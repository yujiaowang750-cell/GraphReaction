"""固定138权重：代表窗口重复评价、processor梯度，以及真实顺序的累积组。无step。"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controlled_experiment.common import (contributions, load_parent, write_json,
    model_hash, tensor_tree_hash)
from full_training_windows import FullTrainingWindows
from training_windows import WindowReader, order_hash
from initialization import create_model
from train import configure_runtime
from loss import compute_loss
from state_output import recover_physical_state
from model_config import FIELD_NAMES, FIELD_STATE_COLUMNS
from training_metrics import metric_values
from run_files import write_csv


def vectors(model, reader, pair):
    data, current, future, target = reader.window("train", *pair)
    prediction = model(data)["prediction"]
    losses = compute_loss(prediction, target, current, data)
    parameters = list(model.processor.parameters())
    parts = contributions(losses)
    result = {}
    for n, (name, part) in enumerate(parts.items()):
        gradients = torch.autograd.grad(part, parameters, retain_graph=n < 3, allow_unused=True)
        # 未被该任务使用的参数贡献为0；零范数的余弦另行标记为未定义。
        result[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                  for p, g in zip(parameters, gradients)]).double().cpu()
    return result


def describe(vectors):
    row = {}
    for name, v in vectors.items():
        row[name + "_finite"] = bool(torch.isfinite(v).all())
        row[name + "_norm"] = float(v.norm()) if row[name + "_finite"] else None
    for a, b in itertools.combinations(vectors, 2):
        na, nb = row[a + "_norm"], row[b + "_norm"]
        row[a + "_" + b + "_cosine"] = float(torch.dot(vectors[a], vectors[b]) / (na * nb)) if na and nb else None
    return row


def distributions(rows):
    result = {}
    for key in rows[0]:
        if not (key.endswith("_norm") or key.endswith("_cosine")):
            continue
        values = np.array([r[key] for r in rows if r[key] is not None])
        result[key] = dict(defined=len(values), undefined=len(rows)-len(values),
            quantiles=dict(zip(("min", "p10", "median", "p90", "max"), np.quantile(values, [0,.1,.5,.9,1]).tolist())) if len(values) else {})
        if key.endswith("_cosine"):
            result[key]["negative_fraction_of_defined"] = float((values < 0).mean()) if len(values) else None
    return result


def repeat_evaluation(model, reader, panels):
    before = model_hash(model)
    model.eval()
    max_prediction = max_loss = max_roundtrip = 0.
    rows = []
    differences = []
    for scope, sid, t in panels:
        data, current, future, target = reader.window(scope, sid, t)
        with torch.no_grad():
            predictions = [model(data)["prediction"] for _ in range(5)]
            losses = [compute_loss(p, target, current, data) for p in predictions]
            recovered, _ = recover_physical_state(current, target, data)
            for index, name in enumerate(FIELD_NAMES):
                a = predictions[0][name]
                for repetition, pred in enumerate(predictions[1:], 1):
                    b = pred[name]
                    error = float((a-b).abs().max())
                    max_prediction = max(max_prediction, error)
                    differences.append(dict(scope=scope,sid=sid,t=t,field=name,repetition=repetition,
                        max_abs_error=error, within_existing_tolerance=bool(torch.allclose(a,b,atol=2e-6,rtol=2e-5))))
                nodes = data.field_nodes[name]
                error = (recovered[nodes, FIELD_STATE_COLUMNS[index]] - future[nodes, FIELD_STATE_COLUMNS[index]]).abs()
                max_roundtrip = max(max_roundtrip, float(error.max()))
            for repetition, loss in enumerate(losses[1:],1):
                error=float((losses[0]["loss"]-loss["loss"]).abs())
                max_loss=max(max_loss,error)
                differences.append(dict(scope=scope,sid=sid,t=t,field="CE2",repetition=repetition,
                    max_abs_error=error, relative_error=error/float(losses[0]["loss"]),
                    within_existing_tolerance=bool(torch.allclose(losses[0]["loss"],loss["loss"],atol=1e-9,rtol=2e-4))))
            rows.append(dict(scope=scope, sid=sid, t=t, **metric_values(losses[0], predictions[0], target, data)))
    assert model_hash(model) == before
    return dict(panels=rows, repetitions=5, differences=differences,
                all_within_existing_tolerance=all(r["within_existing_tolerance"] for r in differences),
                model_and_buffers_unchanged=True, max_prediction_absolute_error=max_prediction,
                max_CE2_absolute_error=max_loss, max_target_physical_roundtrip_error=max_roundtrip,
                prediction_tolerance=[2e-6,2e-5], loss_tolerance=[1e-9,2e-4])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--checkpoint155")
    parser.add_argument("--sha256155")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    configure_runtime(torch.device(args.device))
    dataset = FullTrainingWindows(args.dataset_root)
    saved = load_parent(args.checkpoint, dataset, args.sha256)
    model = create_model(device=args.device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    reader = WindowReader(dataset, args.device)
    before = model_hash(model)
    selection = []
    for sid, sample in dataset.samples.items():
        if sample["split"] != "train":
            continue
        for bin_id, part in enumerate(np.array_split(sorted(sample["input_time_indices"]), 8)):
            selection.append(dict(sid=sid, t=int(part[len(part)//2]), bin=bin_id, trajectory=sample["trajectory_id"]))
    write_csv(out / "selected_520.csv", selection, list(selection[0]))
    assert len(selection) == 520 and len({x["sid"] for x in selection}) == 65
    panels = []
    for scope in ("train", "full"):
        ids = sorted({s for s,t in dataset.pairs(scope)})
        for sid in (ids[0], ids[len(ids)//2], ids[-1]):
            times = dataset.samples[sid]["input_time_indices"]
            for t in (times[0], times[len(times)//2], times[-1]):
                panels.append((scope,sid,t))
    write_json(out / "repeat_evaluation.json", repeat_evaluation(model, reader, panels))
    print("repeat_evaluation recorded: 18 fixed windows, 5 repeats", flush=True)
    cpu_model=create_model(device="cpu")
    cpu_model.load_state_dict(saved["model"],strict=True)
    write_json(out / "repeat_evaluation_cpu.json", repeat_evaluation(cpu_model,WindowReader(dataset,"cpu"),panels[:3]))
    del cpu_model
    write_json(out / "parameter_scope.json", dict(scope="model.processor only; not full model",
        parameters={n:list(p.shape) for n,p in model.processor.named_parameters()},
        numel=sum(p.numel() for p in model.processor.parameters()), model_numel=sum(p.numel() for p in model.parameters())))
    rows=[]
    for i, sample in enumerate(selection,1):
        rows.append({**sample, **describe(vectors(model,reader,(sample["sid"],sample["t"])))})
        if i % 40 == 0:
            print(f"processor gradients {i}/520",flush=True)
    write_csv(out / "window_gradients138.csv", rows, list(rows[0]))
    summaries = {"window138": distributions(rows)}
    groups=[]
    for epoch in (139,155):
        order=dataset.epoch_order(epoch)
        for offset in (0, 4, 1000, 12000, 22924):
            pairs=order[offset:offset+4]
            summed=None
            for pair in pairs:
                g=vectors(model,reader,pair)
                summed = g if summed is None else {k:summed[k]+g[k] for k in g}
            # 先合并向量，再除实际组大小，遵循真实梯度累积；并非平均余弦。
            averaged={k:v/len(pairs) for k,v in summed.items()}
            groups.append(dict(order_epoch=epoch, offset=offset, pairs=pairs, size=len(pairs),
                               order_sha256=order_hash(order), **describe(averaged)))
    write_json(out / "accumulation_groups138.json", groups)
    summaries["accumulation138"]=distributions(groups)
    assert model_hash(model)==before and all(p.grad is None for p in model.parameters())
    if args.checkpoint155:
        saved155=load_parent(args.checkpoint155,dataset,args.sha256155,epoch=155)
        model.load_state_dict(saved155["model"],strict=True)
        before155=model_hash(model)
        rows155=[]
        # 同一组16窗口：首尾训练轨迹，各8时间箱；同时保留138对应结果。
        small=selection[:8]+selection[-8:]
        for sample in small:
            rows155.append({**sample,**describe(vectors(model,reader,(sample["sid"],sample["t"])))})
        write_csv(out / "window_gradients155.csv",rows155,list(rows155[0]))
        summaries["paired_155"]=distributions(rows155)
        keys={(r["sid"],r["t"]) for r in small}
        summaries["paired_138"]=distributions([r for r in rows if (r["sid"],r["t"]) in keys])
        assert model_hash(model)==before155
    write_json(out / "summary.json",dict(status="completed", diagnostics=summaries, optimizer_updates=0,
        model_and_buffers_unchanged=True, autograd_grad_does_not_fill_parameter_grad=True,
        dataset_identity=dataset.identity, checkpoint_sha256=args.sha256, test_dynamic_reads=0))
    print("gradient audit completed; optimizer updates=0",flush=True)


if __name__ == "__main__":
    main()
