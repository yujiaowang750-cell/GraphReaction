"""固定小面板、低频前向的隐藏量诊断；不将 q/a 当作物理电流。"""

import json
from pathlib import Path

import torch

from run_files import write_json, write_csv, read_csv

RATIO_EPS = 1e-12
DIAGNOSTIC_COLUMNS = ["format_version", "run_id", "epoch", "updates", "checkpoint", "sample_index", "input_time_index",
    "scope", "nodes", "physical_edges", "reaction_units", "reaction_incidences", "metric", "step", "group",
    "count", "RMS", "mean", "min", "max", "std", "value", "status"]


def summarize(value):
    if value is None:
        return dict(count=0, RMS=None, mean=None, min=None, max=None, std=None, status="not_computed")
    value = value.detach().double()
    if value.numel() == 0:
        return dict(count=0, RMS=None, mean=None, min=None, max=None, std=None, status="not_applicable")
    if not torch.isfinite(value).all():
        raise FloatingPointError("机制诊断出现非有限隐藏量。")
    return dict(count=value.numel(), RMS=float(value.square().mean().sqrt()), mean=float(value.mean()),
                min=float(value.min()), max=float(value.max()), std=float(value.std(unbiased=False)), status="ok")


def capture_rows(output, data, base):
    trace = output["trace"]
    if trace is None:
        raise ValueError("机制诊断需要同次前向 capture=True。")
    rows = []
    base = dict(base, format_version="mechanism_v1", nodes=data.num_nodes, physical_edges=data.edge_index.shape[1]//2,
                reaction_units=len(data.reaction_center), reaction_incidences=len(data.reaction_node))
    node_groups = {"all": slice(None)}
    node_groups.update({f"region_{name}": data.node_region == r for r, name in enumerate(("NEG", "SEP", "POS"))})
    node_groups.update({f"phase_{p}": data.node_phase == p for p in range(1,8)})
    unit_groups = {"all": slice(None), "reaction_NEG": data.reaction_region == 0, "reaction_POS": data.reaction_region == 1}
    member_region = data.reaction_region[data.reaction_index]
    member_groups = {"all": slice(None), "reaction_NEG": member_region == 0, "reaction_POS": member_region == 1}
    for role, name in enumerate(("CENTER", "ION", "ELECTRON"), 1):
        member_groups[f"role_{name}"] = data.reaction_role == role
        for r, region in enumerate(("NEG", "POS")):
            member_groups[f"{region}_{name}"] = (member_region == r) & (data.reaction_role == role)
    def add(metric, step, values, groups):
        for group, mask in groups.items():
            if len(data.reaction_center) == 0 and metric in ("R_RMS", "q_RMS", "a", "Z_RMS", "DR_RMS", "U_reaction_RMS"):
                selected = values.new_zeros((0,)) if values is not None else None
            else:
                selected = values[mask] if values is not None else None
            rows.append(dict(base, metric=metric, step=step, group=group, value=None,
                             **summarize(selected)))
    for k, h in enumerate(trace["H"]):
        add("H_RMS", k, h, node_groups)
    for k, z in enumerate(trace["Z"]):
        add("Z_RMS", k, z, unit_groups)
    for k, parts in enumerate(trace["steps"]):
        for key, name in (("Dgraph","Dgraph_RMS"), ("R","R_RMS"), ("U","U_node_RMS")):
            add(name, k, parts[key], node_groups)
        add("q_RMS", k, parts["q"], member_groups)
        add("DR_RMS", k, parts["DR"], unit_groups)
        add("U_reaction_RMS", k, parts["UR"], unit_groups)
        for group, mask in node_groups.items():
            r, d = summarize(parts["R"][mask]), summarize(parts["Dgraph"][mask])
            row = dict(base, metric="R_to_Dgraph_RMS_ratio", step=k, group=group, count=r["count"],
                       RMS=None, mean=None, min=None, max=None, std=None, status=r["status"],
                       value=r["RMS"]/(d["RMS"]+RATIO_EPS) if r["count"] else None)
            if len(data.reaction_center) == 0:
                row.update(count=0, value=None, status="not_applicable")
            rows.append(row)
    add("a", -1, trace["a"], member_groups)
    return rows


@torch.no_grad()
def run_diagnostics(model, reader, run, epoch, updates, checkpoint, tag=None):
    run = Path(run)
    name = tag or f"epoch_{epoch:03d}"
    path = run / "diagnostics" / f"capture_{name}.csv"
    if path.exists():
        rows = read_csv(path)
        if rows and rows[0]["checkpoint"] != checkpoint:
            raise ValueError("已有诊断的 checkpoint 身份不一致，不能覆盖。")
        return
    before = model.training
    model.eval()
    rows = []
    for scope, sid, t in reader.dataset.panel:
        data, _, _, _ = reader.window(scope, sid, t)
        output = model(data, capture=True)
        base = dict(run_id=run.name, epoch=epoch, updates=updates, checkpoint=checkpoint,
                    sample_index=sid, input_time_index=t, scope=scope)
        rows.extend(capture_rows(output, data, base))
    model.train(before)
    write_csv(path, rows, DIAGNOSTIC_COLUMNS)
    combined = []
    for source in sorted(path.parent.glob("capture_*.csv")):
        combined.extend(read_csv(source))
    write_csv(path.parent / "mechanism_metrics.csv", combined, DIAGNOSTIC_COLUMNS)


def topology_record(graph):
    degree = graph.reaction_degree
    covered = degree > 0
    groups = {}
    for name, mask in [(f"phase{p}", graph.node_phase == p) for p in range(1,8)] + [
            (name, graph.node_region == r) for r,name in enumerate(("NEG","SEP","POS"))]:
        count, hits = int(mask.sum()), int(covered[mask].sum())
        groups[name] = dict(nodes=count, reaction_covered_nodes=hits, reaction_coverage=hits/count if count else None)
    histogram = torch.bincount(degree).tolist()
    return dict(nodes=graph.num_nodes, physical_edges=graph.edge_index.shape[1]//2,
                directed_arcs=graph.edge_index.shape[1], reaction_units=len(graph.reaction_center),
                reaction_incidences=len(graph.reaction_node), physical_group_directed_counts=torch.bincount(graph.edge_group,minlength=5).tolist(),
                groups=groups, no_reaction_nodes=int((~covered).sum()), no_reaction_fraction=float((~covered).float().mean()),
                reaction_degree_histogram=histogram, zero_physical_indegree_nodes=int((graph.graph_degree == 0).sum()))


class TopologyCoverage:
    def __init__(self, run, identity):
        self.path = Path(run) / "diagnostics/topology_coverage.json"
        self.identity = identity
        self.graphs = json.loads(self.path.read_text(encoding="utf-8"))["graphs"] if self.path.exists() else {}

    def add(self, sid, graph):
        if str(sid) not in self.graphs:
            self.graphs[str(sid)] = topology_record(graph)
            write_json(self.path, dict(format_version="topology_coverage_v1", identity=self.identity, graphs=self.graphs))
