"""指标、日志无扰动、精确恢复和文件契约；人工图与少量真实窗口检查。"""

import ast
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_helpers import HERE, reference_definitions, small_graph, must_fail
from data_loader import load_window, load_static_graph
from diagnostics import capture_rows, summarize
from evaluation import physical_error_arrays
from initialization import create_model
from loss import compute_loss
from model_config import FIELD_NAMES
from state_output import build_delta_targets
from training_metrics import metric_values, MetricMean, ErrorSums
from training_windows import TrainingWindows, WindowReader, order_hash
from training_step import train_group
from train import run_training, new_optimizer, configure_runtime
from training_checkpoint import rng_state, reconcile_csv
from run_files import read_csv, write_json, file_hash


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for name in a:
            assert_nested_equal(a[name], b[name])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b):
            assert_nested_equal(x,y)
    else:
        assert a == b, (a,b)


def scalar_tests(ref, report):
    graph, current, future, _, _ = small_graph(ref)
    target = build_delta_targets(current, future, graph)
    prediction = {name: value.float().clone() for name,value in target.items()}
    # 双精度人工预测避免 float32 表示误差影响可手算断言。
    prediction = {name: value.clone() for name,value in target.items()}
    for j, field in enumerate(FIELD_NAMES):
        prediction[field] += j+1
    field = FIELD_NAMES[0]
    phase = graph.node_phase[graph.field_nodes[field]]
    prediction[field] = target[field] + torch.where(phase == 2, 2., 4.)
    losses = compute_loss(prediction, target, current, graph)
    metrics = metric_values(losses, prediction, target, graph)
    np.testing.assert_allclose([metrics["ce_phase2_MSE"],metrics["ce_phase5_MSE"],metrics["ce_MSE"]],
                               [4,16,10],rtol=0,atol=1e-12)
    expected_fields = np.array([10,4,9,16,25,36,49], dtype=float)
    expected_ce2 = (2*10+expected_fields[1:].sum())/8 + .1*float(losses["current_loss"])
    expected_q = expected_fields.mean() + .1*float(losses["current_loss"])
    np.testing.assert_allclose([metrics["CE2"],metrics["Q_ref"]], [expected_ce2,expected_q], rtol=0, atol=1e-12)
    report["loss_metric_max_error"] = float(max(abs(metrics["CE2"]-expected_ce2),abs(metrics["Q_ref"]-expected_q)))
    mean = MetricMean()
    one = {k:1. for k in mean.sums}; nine = {k:9. for k in mean.sums}
    mean.add(one,1); mean.add(nine,3)
    assert mean.result()["CE2"] == 7
    np.testing.assert_allclose(mean.result()["current_relative_RMSE_pct"],100*np.sqrt(7),rtol=0,atol=1e-12)
    online = MetricMean(); online.add(one,1,denominator=1); online.add(nine,3,denominator=1)
    assert online.result()["CE2"] == 14
    errors = ErrorSums(); errors.add(np.array([1.,1.]),np.array([1.,1.]),1)
    errors.add(np.array([3.]),np.array([3.]),4)
    result = errors.result()
    assert result["count"] == 3 and result["weighted_count"] == 6
    np.testing.assert_allclose([result["physical_RMSE"],result["physical_MAE"]],[np.sqrt(38/6),14/6],rtol=0,atol=1e-12)
    empty = ErrorSums().result()
    assert empty["physical_RMSE"] is None and empty["count"] == 0
    # cmax=100/250，只恢复一次；预期浓度误差为3/7.5 mol/m^3。
    graph.delta_mean[2:4] = graph.concentration_scale[2:4] * .01
    graph.delta_std[2:4] = graph.concentration_scale[2:4] * .02
    future = current.clone()
    target = build_delta_targets(current, future, graph)
    prediction = {name: torch.zeros_like(value) for name,value in target.items()}
    for j in (2,3,4): prediction[FIELD_NAMES[j]].fill_(1)
    _, physical = physical_error_arrays(prediction,target,current,future,graph)
    np.testing.assert_allclose(physical["cg"][0],3,rtol=0,atol=1e-12)
    np.testing.assert_allclose(physical["cp"][0],7.5,rtol=0,atol=1e-12)
    np.testing.assert_allclose(physical["phi_p"][0],1,rtol=0,atol=1e-10)
    report["physical_units"] = "PASS: cg=3, cp=7.5 mol/m^3; phi_p=1 mV; cmax and mV applied once"


class ArtificialDataset:
    def __init__(self, ref):
        self.graph, self.current, self.future, _, _ = small_graph(ref)
        self.identity = {"artificial_fixture": "training_metrics_v1"}
        self.proxy_root = self.root = self.full_root = Path("artificial_only")
        self.rows = {scope:[dict(sid=sid,t=t) for t in range(count)] for scope,sid,count in
                     (("train",1,8),("proxy",0,4),("full",0,6))}
        self.by_pair = {s:{(r["sid"],r["t"]):r for r in rows} for s,rows in self.rows.items()}
        self.bins = {(r["sid"],r["t"]):r["t"]%8 for rows in self.rows.values() for r in rows}
        self.panel = [("train",1,0),("proxy",0,0)]

    epoch_order = TrainingWindows.epoch_order
    pairs = TrainingWindows.pairs

    def weight(self, scope, sid, t):
        return 1. if scope == "full" else .5 if t%2 == 0 else 1.5


class ArtificialReader:
    def __init__(self, dataset, device):
        self.dataset, self.device, self.on_graph = dataset, device, None
        self.cpu_graph = dataset.graph

    def window(self, scope, sid, t):
        assert (sid,t) in self.dataset.by_pair[scope]
        data = self.dataset.graph.clone().to(self.device)
        data.input_time_index = t
        current, future = self.dataset.current.to(self.device), self.dataset.future.to(self.device)
        if self.on_graph: self.on_graph(sid,self.cpu_graph)
        return data,current,future,build_delta_targets(current,future,data)


def logging_and_diagnostics(ref, report):
    dataset = ArtificialDataset(ref)
    reader = ArtificialReader(dataset,"cpu")
    first, second = create_model(), create_model()
    opt1, opt2 = new_optimizer(first), new_optimizer(second)
    calls = [0,0]
    hooks = [first.register_forward_hook(lambda *a: calls.__setitem__(0,calls[0]+1)),
             second.register_forward_hook(lambda *a: calls.__setitem__(1,calls[1]+1))]
    before = rng_state()
    values,norm1 = train_group(first,reader,dataset.pairs("train")[:4],opt1,True)
    _,norm2 = train_group(second,reader,dataset.pairs("train")[:4],opt2,False)
    assert calls == [4,4] and norm1 == norm2
    for hook in hooks: hook.remove()
    assert_nested_equal(first.state_dict(),second.state_dict())
    assert_nested_equal(opt1.state_dict(),opt2.state_dict())
    assert_nested_equal(before,rng_state())
    report["logging_parameter_max_error"] = 0.
    report["logging_optimizer_and_rng"] = "PASS: one forward/backward per training window, identical updates and RNG"
    report["pre_clip_norm"] = norm1
    assert norm1 > 1  # 确实触发裁剪，而不是只测未裁剪路径。
    for mode in ("overlap","no_reaction"):
        data,_,_,_,_ = small_graph(ref,mode)
        for full in (False,True):
            with torch.no_grad(): output = first(data,capture=True,compute_terminal_reaction=full)
            rows = capture_rows(output,data,dict(run_id="test",epoch=1,updates=1,checkpoint="test",
                                sample_index=1,input_time_index=0,scope="train"))
            assert {r["step"] for r in rows if r["metric"]=="H_RMS"} == set(range(5))
            assert {r["step"] for r in rows if r["metric"]=="Z_RMS"} == set(range(5 if full else 4))
            if not full:
                assert all(r["RMS"] is None and r["status"]=="not_computed" for r in rows if r["metric"]=="DR_RMS" and r["step"]==3)
            if mode == "no_reaction":
                assert all(r["status"]=="not_applicable" for r in rows if r["metric"] in ("a","q_RMS","R_RMS"))
            json.dumps(rows,allow_nan=False)
    assert summarize(torch.empty(0))["RMS"] is None
    try: summarize(torch.tensor([float("nan")]))
    except FloatingPointError: pass
    else: raise AssertionError("非有限诊断未报错")
    report["mechanism"] = "PASS: normal/empty/terminal diagnostic, regions and roles, nonfinite rejection"


def resume_tests(ref, report, output_root):
    dataset = ArtificialDataset(ref)
    continuous, resumed = output_root / "continuous", output_root / "resumed"
    with patch("train.WindowReader",ArtificialReader):
        run_training(continuous,dataset,epochs=3,device="cpu",smoke_windows=8)
        # 先停在epoch中间，再停在epoch边界，再继续两个epoch。
        run_training(resumed,dataset,epochs=3,device="cpu",smoke_windows=8,pause_after_updates=1)
        # 模拟 checkpoint 已提交，但最后一条CSV尚未落盘。
        (resumed / "update_metrics.csv").unlink()
        run_training(resumed,dataset,epochs=3,device="cpu",smoke_windows=8,resume=True,pause_after_epoch=1)
        run_training(resumed,dataset,epochs=3,device="cpu",smoke_windows=8,resume=True)
    one = torch.load(continuous/"checkpoints/last.pt",map_location="cpu",weights_only=True)
    two = torch.load(resumed/"checkpoints/last.pt",map_location="cpu",weights_only=True)
    for name in ("model","optimizer","scheduler","rng"):
        assert_nested_equal(one[name],two[name])
    for name in ("epoch","cursor","updates","best_epoch","best_train_CE2","early_stop_counter","online"):
        assert_nested_equal(one["state"][name],two["state"][name])
    updates = read_csv(resumed/"update_metrics.csv")
    assert [int(row["update"]) for row in updates] == list(range(1,7))
    assert [int(row["epoch"]) for row in read_csv(resumed/"history.csv")] == [1,2,3]
    for left,right in zip(read_csv(continuous/"history.csv"),read_csv(resumed/"history.csv")):
        for key in left:
            if "seconds" not in key: assert left[key] == right[key], key
    for scope in ("train","proxy","full"):
        folder = resumed / "evaluation" / scope
        assert json.loads((folder/"overall_metrics.json").read_text(encoding="utf-8"))["status"] == "completed"
        for filename in ("field_metrics.csv","region_metrics.csv","stage_metrics.csv","voltage_current.csv"):
            assert read_csv(folder/filename)
    report["resume_parameter_max_error"] = 0.
    report["resume"] = "PASS: mid-epoch + epoch boundary + two further epochs; recover one missing CSV row; no duplicate IDs"
    report["artificial_example_run"] = str(resumed)


def real_data_tests(report):
    dataset = TrainingWindows()
    # 直接从旧文件提取纯window_order函数，避免导入历史runtime副作用。
    path = Path(r"D:\Github Project\subdatasets\processed_v1\reproduction_proxy\reference\a0_source\train.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="window_order")
    environment = {"np":np}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(path),"exec"),environment)
    before = np.random.get_state()
    np.random.seed(2026)
    try:
        for epoch in range(1,4):
            old = environment["window_order"](dataset,"train",True)
            assert old == dataset.epoch_order(epoch)
            assert order_hash(old) == order_hash(dataset.epoch_order(epoch))
    finally: np.random.set_state(before)
    reader = WindowReader(dataset,"cpu")
    sid,t = dataset.pairs("train")[0]
    data,current,future,target = reader.window("train",sid,t)
    original,current2,future2 = load_window(reader.folder,t,dataset.statistics,reader.cpu_graph)
    torch.testing.assert_close(data.x,original.x,rtol=0,atol=0)
    torch.testing.assert_close(current,current2,rtol=0,atol=0)
    torch.testing.assert_close(future,future2,rtol=0,atol=0)
    must_fail(lambda: reader.window("train",*dataset.pairs("proxy")[0]),"禁止读取")
    report["data_and_order"] = "PASS: 65/2080, 14/448, 14/4917; original 3-epoch order exactly matches; cached reader exact"
    report["source_identity"] = dataset.identity


def main():
    configure_runtime("cpu")
    ref = reference_definitions()
    report = dict(status="running", formal_training_started=False)
    scalar_tests(ref,report)
    logging_and_diagnostics(ref,report)
    real_data_tests(report)
    number=1
    while (HERE/"training_runs"/f"metrics_checks_{number:03d}").exists(): number+=1
    output = HERE/"training_runs"/f"metrics_checks_{number:03d}"
    output.mkdir(parents=True)
    resume_tests(ref,report,output)
    report.update(status="PASS", source_sha256={p.name:file_hash(p) for p in HERE.glob("*.py") if p.name!="model_guide.py"})
    write_json(output/"test_results.json",report)
    print("METRICS_CHECKS_PASS",output,flush=True)


if __name__=="__main__": main()
