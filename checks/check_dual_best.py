"""新全量流程的必要检查：验证无更新、20轮早停、双best、断点恢复。"""

import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from controlled_experiment.common import tensor_tree_hash
from epoch_validation import validate_full, update_validation_best
from full_training_windows import FullTrainingWindows
from initialization import create_model
from run_files import file_hash, write_json
from train import initial_state, configure_runtime, model_hash, run_training
from training_checkpoint import rng_state
from training_checkpoint import load_training_checkpoint, archive_epoch_checkpoint
from train import new_optimizer
from learning_rate import create_scheduler
from group_metrics import group_metrics
from training_metrics import SHORT_FIELDS
from training_config import SEED, EARLY_STOPPING
from training_windows import WindowReader


def compare_tensors(a, b):
    errors = []
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)
        errors.append(float((a-b).abs().max()) if a.numel() else 0.)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: errors.append(compare_tensors(a[key],b[key]))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): errors.append(compare_tensors(x,y))
    else: assert a==b,(a,b)
    return max(errors,default=0.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    configure_runtime('cpu')
    dataset=FullTrainingWindows(args.dataset_root)
    protected={str(dataset.root/name):file_hash(dataset.root/name) for name in dataset.identity['source']}
    for run_name in ('full_train_ce2_20261026_215337','full_continue_193_fixed_lr_20261030_101435'):
        for filename in ('best.pt','last.pt'):
            path=HERE/'training_runs'/run_name/'checkpoints'/filename
            if path.exists(): protected[str(path)]=file_hash(path)
    state=initial_state();assert EARLY_STOPPING==20
    assert update_validation_best(state,{'CE2':.5,'Q_ref':.6},1,1)
    for epoch in range(2,22):
        # 即使训练best逐轮下降，验证相等/变差仍连续累计20轮。
        state['best_train_CE2']=1./epoch
        update_validation_best(state,{'CE2':.5 if epoch%2 else .6,'Q_ref':.7},epoch,epoch)
        assert state['early_stop_counter']==epoch-1
        assert (state['early_stop_counter']>=20)==(epoch==21)
    assert state['best_val_epoch']==1
    assert update_validation_best(state,{'CE2':.5-1e-12,'Q_ref':.6},22,22)
    assert state['early_stop_counter']==0 and state['best_val_epoch']==22
    old=tensor_tree_hash(state)
    try: update_validation_best(state,{'CE2':float('nan'),'Q_ref':.1},23,23)
    except FloatingPointError: pass
    else: raise AssertionError('NaN validation must fail')
    assert tensor_tree_hash(state)==old

    model=create_model(SEED,'cpu');model.train()
    reader=WindowReader(dataset,'cpu');read_pairs=[]
    original_window=reader.window
    def tracked_window(scope,sid,t):
        read_pairs.append((scope,sid,t));return original_window(scope,sid,t)
    reader.window=tracked_window
    before=model_hash(model);random_before=tensor_tree_hash(rng_state())
    calls=[]
    def before_forward(module,inputs):
        assert not module.training and not torch.is_grad_enabled();calls.append(1)
    hook=model.register_forward_pre_hook(before_forward)
    metrics=validate_full(model,reader,smoke_windows=2);hook.remove()
    assert read_pairs==[('full',sid,t) for sid,t in dataset.pairs('full')[:2]]
    assert metrics.mean.windows==2 and len(calls)==2
    assert model_hash(model)==before and model.training and all(p.grad is None for p in model.parameters())
    assert tensor_tree_hash(rng_state())==random_before

    continuous=out/'continuous';resumed=out/'resumed'
    run_training(continuous,dataset,epochs=3,device='cpu',smoke_windows=2)
    run_training(resumed,dataset,epochs=3,device='cpu',smoke_windows=2,pause_after_epoch=1)
    # 模拟last已提交但验证best副本尚未提交；恢复应从last补齐。
    missing=resumed/'checkpoints/best_val.pt'
    assert missing.resolve().is_relative_to(out)
    missing.unlink()
    missing_epoch = resumed/'checkpoints/epoch_0001.pt'
    missing_epoch.unlink()  # 仅删除本次检查目录的副本，模拟归档前中断。
    run_training(resumed,dataset,epochs=3,device='cpu',smoke_windows=2,resume=True,pause_after_updates=2)
    assert missing.exists() and missing_epoch.exists()
    # 第2轮训练已完成、验证尚未开始时暂停；恢复仅补验证，不再更新该轮。
    run_training(resumed,dataset,epochs=3,device='cpu',smoke_windows=2,resume=True)
    one=torch.load(continuous/'checkpoints/last.pt',map_location='cpu',weights_only=True)
    two=torch.load(resumed/'checkpoints/last.pt',map_location='cpu',weights_only=True)
    errors={name:compare_tensors(one[name],two[name]) for name in ('model','optimizer','scheduler','rng')}
    assert one['state']['updates']==two['state']['updates']==3
    for folder,payload in ((continuous,one),(resumed,two)):
        history=payload['state']['history']
        train_best=min(history,key=lambda x:x['online_train_CE2'])['epoch']
        val_best=min(history,key=lambda x:x['val_CE2'])['epoch']
        for filename,epoch in (('best_train.pt',train_best),('best_val.pt',val_best)):
            checkpoint=torch.load(folder/'checkpoints'/filename,map_location='cpu',weights_only=True)
            assert checkpoint['state']['epoch']==epoch
        assert payload['state']['early_stop_counter']==3-val_best
        assert all(h['validation_windows']==2 and h['early_stopping_monitor']=='val_CE2' for h in history)
        summary=json.loads((folder/'training_summary.json').read_text(encoding='utf-8'))
        assert summary['final_evaluation_checkpoint']=='best_val'
        evaluated=json.loads((folder/'evaluation/full/overall_metrics.json').read_text(encoding='utf-8'))
        assert evaluated['checkpoint']['epoch']==val_best and Path(evaluated['checkpoint']['path']).name=='best_val.pt'
        assert len(list((folder/'checkpoints').glob('epoch_*.pt')))==3
        for epoch in range(1,4):
            archived=torch.load(folder/f'checkpoints/epoch_{epoch:04d}.pt',map_location='cpu',weights_only=True)
            assert archived['state']['epoch']==epoch and archived['state']['cursor']==0
            assert len(archived['state']['history'])==epoch
            assert archived['state']['updates']==epoch
            assert archived['scheduler']['last_epoch']==epoch
            assert archived['pending_gradients'] is None and archived['accumulation_pending_windows']==0
            assert all(float(s['step'])==epoch for s in archived['optimizer']['state'].values())
            assert all(key in archived for key in ('model','optimizer','scheduler','rng','initialization','identity'))
        for row in history:
            for prefix in ('train','val'):
                values={key[len(prefix)+1:]:value for key,value in row.items() if key.startswith(prefix+'_')}
                assert all(values[field+'_MSE'] is not None for field in SHORT_FIELDS)
                grouped=group_metrics(values)
                for key,value in grouped.items(): assert values[key]==value
                total=sum(grouped[f'{g}_CE2_contribution'] for g in ('C','P','V','I'))
                assert abs(total-values['CE2']) <= 1e-12*max(1.,abs(values['CE2']))
        assert len(set(history[0]))==len(history[0])
        assert (folder/'metrics_v2/group_loss_curves.png').exists()

    # 每轮归档可加载完整训练状态，不能只有模型权重。
    archived_path=resumed/'checkpoints/epoch_0001.pt'
    restored_model=create_model(SEED,'cpu');restored_optimizer=new_optimizer(restored_model)
    restored_scheduler=create_scheduler(restored_optimizer)
    saved=load_training_checkpoint(archived_path,restored_model,restored_optimizer,restored_scheduler,two['identity'])
    assert tensor_tree_hash(restored_model.state_dict())==tensor_tree_hash(saved['model'])
    assert tensor_tree_hash(restored_optimizer.state_dict())==tensor_tree_hash(saved['optimizer'])
    assert tensor_tree_hash(restored_scheduler.state_dict())==tensor_tree_hash(saved['scheduler'])
    assert tensor_tree_hash(rng_state())==tensor_tree_hash(saved['rng'])
    # 归档已存在且内容不同应报错，而不是覆盖旧轮权重。
    before_archive=file_hash(archived_path)
    try: archive_epoch_checkpoint(resumed/'checkpoints/last.pt',1)
    except ValueError: pass
    else: raise AssertionError('existing epoch checkpoint was overwritten')
    assert file_hash(archived_path)==before_archive
    # 新实验误用已有目录必须拒绝；不会重置历史。
    before_last=file_hash(resumed/'checkpoints/last.pt')
    try: run_training(resumed,dataset,epochs=3,device='cpu',smoke_windows=2)
    except FileExistsError: pass
    else: raise AssertionError('existing run accepted as new experiment')
    assert file_hash(resumed/'checkpoints/last.pt')==before_last
    assert all(file_hash(path)==sha for path,sha in protected.items())
    write_json(out/'checks.json',dict(status='passed',seed=SEED,patience=20,min_delta=0,
        ties_count_as_no_improvement=True,tiny_strict_improvement_resets=True,train_best_does_not_reset_validation_counter=True,
        validation_no_parameter_or_rng_update=True,validation_uses_full_scope=True,
        double_best_and_missing_copy_recovery=True,mid_epoch_and_epoch_resume=True,
        full_epoch_archives=True,missing_epoch_archive_recovery=True,
        archived_model_optimizer_scheduler_rng_restore=True,
        existing_run_and_epoch_archive_not_overwritten=True,seven_fields_and_group_metrics=True,
        continuous_vs_resume_max_errors=errors,tolerance=dict(atol=2e-5,rtol=2e-4),
        continuous_updates=3,resumed_updates=3,formal_training_started=False,
        old_checkpoints_and_dataset_identity_files_unchanged=True,test_dynamic_reads=0))
    print('DUAL_BEST_CHECKS_PASSED',out,flush=True)


if __name__=='__main__':main()
