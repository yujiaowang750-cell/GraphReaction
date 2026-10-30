"""只运行人工图和历史记录后处理；不启动正式训练或重做真实数据评价。"""

import json
import math
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v2_helpers import reference_definitions, small_graph
from check_training_metrics import (scalar_tests, logging_and_diagnostics, resume_tests,
                                    ArtificialDataset, ArtificialReader, assert_nested_equal)
from check_train_only import early_stop_test
from check_full_training import check_full_flow
from metric_reporting import EpochMetrics, fields_from_saved, sqrt_mse
from model_config import FIELD_NAMES
from loss import compute_loss
from state_output import build_delta_targets
from training_metrics import BASE_METRICS, SHORT_FIELDS, metric_values
from postprocess_metrics import build_epoch_rows, summarize_selection, process_run
from run_files import read_csv, write_json
from plot_metrics import values
from train import configure_runtime


def node_pooling_test(report):
    totals = EpochMetrics()
    expected = {field: [] for field in SHORT_FIELDS}
    # 不等节点数；第一个窗口可理解为phase2有3个节点、phase5有1个节点。
    for window, ce in enumerate(([1., 1., 1., 5.], [3., 3.])):
        prediction, target = {}, {}
        for j, (field, short) in enumerate(zip(FIELD_NAMES, SHORT_FIELDS)):
            error = torch.tensor(ce if j == 0 else [j+window, j+2*window], dtype=torch.float64)
            prediction[field], target[field] = error, torch.zeros_like(error)
            expected[short].extend(error.tolist())
        metrics = {name: 2. if window == 0 else 10. for name in BASE_METRICS}
        group = EpochMetrics()
        group.add(metrics, prediction, target)
        totals.merge(EpochMetrics(group.state()))
    assert totals.mean.result()["CE2"] == 6  # 两窗口算术平均，没有再次除4。
    assert totals.mean.windows == 2
    max_error = 0.
    for field, numbers in expected.items():
        direct = math.sqrt(sum(x*x for x in numbers) / len(numbers))
        actual = totals.fields()[field]["NRMSE_delta"]
        max_error = max(max_error, abs(actual-direct))
        np.testing.assert_allclose(actual, direct, atol=1e-12, rtol=0)
    pooled = totals.fields()["ce"]
    assert pooled["count"] == 6
    np.testing.assert_allclose(pooled["standardized_node_pooled_MSE"], 46/6, atol=1e-12, rtol=0)
    assert not math.isclose(pooled["NRMSE_delta"], math.sqrt((13+9)/2))
    assert EpochMetrics().fields()["ce"]["NRMSE_delta"] is None
    assert sqrt_mse(0) == 0
    for invalid in (-1, float('nan'), float('inf')):
        try: sqrt_mse(invalid)
        except ValueError: pass
        else: raise AssertionError("无效MSE未拒绝")
    try: fields_from_saved(dict(weight_rule="population_weight"))
    except ValueError: pass
    else: raise AssertionError("不得将加权旧指标冒充不加权NRMSE")
    saved = dict(weight_rule="1", physical_node_pooled={field:dict(
        standardized_node_pooled_MSE=4., count=2, weighted_count=2.) for field in SHORT_FIELDS})
    assert fields_from_saved(saved)['ce']['NRMSE_delta'] == 2.
    saved['physical_node_pooled']['ce']['weighted_count'] = 3.
    try: fields_from_saved(saved)
    except ValueError: pass
    else: raise AssertionError("矛盾的节点计数必须拒绝")
    report["node_pooled_nrmse_max_error"] = max_error
    report["aggregation"] = "PASS: unequal node counts, arithmetic window mean, one final square root, empty/zero/nonfinite cases"


def both_electrolyte_fields_test(ref, report):
    graph, current, future, _, _ = small_graph(ref)
    target = build_delta_targets(current, future, graph)
    prediction = {key: value.clone() for key,value in target.items()}
    for j in (0,1):
        field = FIELD_NAMES[j]
        phase = graph.node_phase[graph.field_nodes[field]]
        prediction[field] += torch.where(phase == 2, 2., 6.)
    losses = compute_loss(prediction,target,current,graph)
    metrics = metric_values(losses,prediction,target,graph)
    np.testing.assert_allclose([metrics['ce_MSE'], metrics['phi_e_MSE']], [20,20], rtol=0, atol=1e-12)
    report["electrolyte_both_fields"] = "PASS: phase2/5 both retain 0.5/0.5"


def selection_and_missing_test(report):
    history = [dict(epoch=i, online_train_CE2=ce, online_train_Q_ref=q)
               for i,ce,q in ((1,.3,.1),(2,.2,.4),(3,.25,.05))]
    observed = {name: 1. for name in BASE_METRICS}
    observed.update(CE2=.45, Q_ref=.5)
    evaluation = dict(scope="full",checkpoint=dict(epoch=2))
    rows=build_epoch_rows(history,dict(train_weight=1,dataset_mode="full"),[(evaluation,observed)])
    assert [r['val_CE2'] for r in rows] == [None,.45,None]
    assert np.isfinite(values(rows,'val_CE2')).tolist() == [False,True,False]
    selected=summarize_selection(history,rows,dict(best_epoch=2,best_train_CE2=.2,selection_metric='online_train_CE2'))
    assert selected['best_epoch']==2 and selected['best_Q_ref_at_CE2_best']==.4
    assert selected['Q_ref_min_diagnostic']==.05 and selected['Q_ref_min_epoch_diagnostic']==3
    assert selected['validation_selected_best_epoch'] is None
    legacy=build_epoch_rows(history,dict(train_weight='mean_loss_weight'),[])
    assert all(r['train_CE2'] is None for r in legacy)
    report['selection']="PASS: CE2-only selection; associated Q_ref is not Q_ref minimum; missing validation stays missing"


def main():
    configure_runtime('cpu')
    report=dict(status='running',formal_training=False,real_data_evaluation=False)
    ref=reference_definitions()
    scalar_tests(ref,report)
    both_electrolyte_fields_test(ref,report)
    node_pooling_test(report)
    selection_and_missing_test(report)
    logging_and_diagnostics(ref,report)
    with tempfile.TemporaryDirectory(prefix='graphreaction_metric_checks_') as tmp:
        folder=Path(tmp)
        # 复用小图执行3轮、每轮8窗口的断点恢复对拍，不是正式训练。
        resume_tests(ref,report,folder)
        a=torch.load(folder/'continuous/checkpoints/last.pt',map_location='cpu',weights_only=True)
        b=torch.load(folder/'resumed/checkpoints/last.pt',map_location='cpu',weights_only=True)
        assert_nested_equal(a['state']['reported'],b['state']['reported'])
        assert a['state']['best_Q_ref_at_CE2_best']==b['state']['best_Q_ref_at_CE2_best']
        report['reported_resume']='PASS: node SSE/count and arithmetic means survive mid-epoch resume'
        dataset=ArtificialDataset(ref)
        dataset.mode='full'
        dataset.weight=lambda scope,sid,t: 1.
        dataset.panel=[('train',1,0),('full',0,0)]
        check_full_flow(dataset,ArtificialReader,folder,'cpu',report,'artificial_full',epochs=2,windows=5)
        path=folder/'artificial_full'
        history=read_csv(path/'history.csv')
        for row in history:
            assert row['train_CE2']==row['online_train_CE2']
            assert row['train_Q_ref']==row['online_train_Q_ref']
            assert row['val_CE2']=='' and row['val_Q_ref']==''
            for field in SHORT_FIELDS:
                assert math.isclose(float(row[f'train_{field}_NRMSE_delta'])**2,
                                    float(row[f'train_{field}_node_pooled_MSE']),rel_tol=1e-12)
        assert (path/'evaluation/full/nrmse_delta.json').exists()
        early_stop_test(ref,folder,report)
    report.pop('artificial_example_run',None)
    run=HERE/'training_runs/full_train_ce2_20261026_215337'
    available=process_run(run)
    assert available['train_epochs']==141
    assert available['validation_epochs']==[138]
    assert len(available['missing_validation_epochs'])==140
    assert available['selection']['best_epoch']==138
    derived=read_csv(run/'metrics_v2/epoch_metrics.csv')
    point=next(r for r in derived if int(r['epoch'])==138)
    assert float(point['val_ce_NRMSE_delta']) > 0
    assert all(not r['train_ce_NRMSE_delta'] for r in derived)
    assert all(not r['val_ce_NRMSE_delta'] for r in derived if int(r['epoch'])!=138)
    report['historical_postprocessing']=dict(train_epochs=141,validation_epochs=[138],
        original_files_unchanged=True,missing_validation_epochs=140,train_nrmse_epochs=0)
    report['status']='PASS'
    write_json(HERE/'checks/metric_reporting_result.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
