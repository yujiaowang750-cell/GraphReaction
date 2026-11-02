"""可移植的独立A/B分支。check不更新；smoke只跑7窗口；train才执行10轮。"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controlled_experiment.common import (ROOT, read_json, write_json, load_parent, initialize_branch,
    objective, grouped, tensor_tree_hash, model_hash)
from full_training_windows import FullTrainingWindows
from training_windows import WindowReader, order_hash
from training_checkpoint import rng_state, restore_rng
from training_metrics import MetricMean, metric_values
from loss import compute_loss
from train import configure_runtime
from evaluation import evaluate_scope
from run_files import file_hash, write_csv, copy_atomic

SCHEMA='graphreaction_fixed8_beta_branch_v1'


def train_group(model,reader,pairs,optimizer,beta):
    """每组清零，按真实组大小反向；结束后只裁剪/更新一次。"""
    model.train(); optimizer.zero_grad(set_to_none=True)
    mean=MetricMean(); objectives=[]
    for sid,t in pairs:
        data,current,future,target=reader.window('train',sid,t)
        prediction=model(data)['prediction']
        losses=compute_loss(prediction,target,current,data)
        target_loss=objective(losses,beta)
        if not torch.isfinite(target_loss): raise FloatingPointError(f'nonfinite loss: {sid},{t}')
        mean.add(metric_values(losses,prediction,target,data))
        objectives.append(float(target_loss.detach()))
        (target_loss/len(pairs)).backward()
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    optimizer.step()
    return mean,objectives,norm


def save_checkpoint(path,model,optimizer,state,identity):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    torch.save(dict(schema=SCHEMA,identity=identity,state=state,model=model.state_dict(),
        optimizer=optimizer.state_dict(),rng=rng_state(),initialization=model.initialization_manifest,
        pending_gradients=None,accumulation_pending_windows=0),temp)
    temp.replace(path)


def evaluate_epoch(model,reader,folder,checkpoint,identity,pairs_by_scope=None):
    """评价不改变模型/缓冲区，且保存恢复RNG，不污染下一轮训练。"""
    before=model_hash(model); random_before=rng_state()
    output={}
    try:
        for scope in ('train','full'):
            target=folder/scope
            if (target/'evaluation_complete.json').exists():
                result=read_json(target/'overall_metrics.json')
                if result['checkpoint']!=checkpoint or result['identity']!=identity:
                    raise ValueError('已有评价身份不一致，拒绝混用。')
            else:
                result=evaluate_scope(model,reader,scope,target,identity,checkpoint,
                    pairs=None if pairs_by_scope is None else pairs_by_scope[scope])
                write_json(target/'evaluation_complete.json',dict(checkpoint=checkpoint,status='completed'))
            output[scope]=grouped(result['losses'])
        if model_hash(model)!=before: raise RuntimeError('评价修改了模型或缓冲区。')
    finally:
        restore_rng(random_before)
    return output


def verify_runtime():
    manifest=read_json(Path(__file__).with_name('runtime_manifest.json'))
    for name,expected in manifest['files'].items():
        if file_hash(ROOT/name)!=expected:
            raise ValueError(f'源码身份不符：{name}；请使用同一上传包，或审阅后建立新的独立版本。')
    return manifest


def run(args):
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('没有可用CUDA；先安装匹配服务器驱动的PyTorch，不静默改为CPU。')
    configure_runtime(torch.device(args.device))
    config=read_json(args.config)
    if config['schema']!=SCHEMA or config['epochs']!=10 or config['lr']!=1e-5 or config['denominator']!=8:
        raise ValueError('本版本仅实现约定的固定分母8、固定学习率1e-5、10轮对照。')
    if config['selection']!='online_original_train_CE2':
        raise ValueError('正式best必须按在线训练的原始CE2选择。')
    beta=float(config['beta'])
    if not math.isfinite(beta) or beta<0: raise ValueError('beta必须有限且非负。')
    runtime=verify_runtime()
    dataset=FullTrainingWindows(args.dataset_root)
    parent=load_parent(args.parent_checkpoint,dataset,config['parent_sha256'])
    model,optimizer=initialize_branch(parent,args.device)
    identity=dict(schema=SCHEMA,config=config,dataset=dataset.identity,source=runtime,
        parent_sha256=config['parent_sha256'],mode=args.mode,
        numerical=dict(torch=str(torch.__version__),numpy=np.__version__,device=args.device,
            device_name=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
            amp=False,tf32=False,deterministic_algorithms=False))
    out=Path(args.output).resolve()
    # 输出必须独立。只有本入口生成、身份一致的目录才允许resume。
    if args.resume:
        if args.mode=='check': raise ValueError('check不需要resume。')
        old=read_json(out/'config.json')
        if old!=identity: raise ValueError('resume配置、源码、设备或数据身份不一致。')
        saved=torch.load(out/'checkpoints/last.pt',map_location='cpu',weights_only=True)
        if saved['schema']!=SCHEMA or saved['identity']!=identity: raise ValueError('分支checkpoint身份错误。')
        if saved['pending_gradients'] is not None or saved['accumulation_pending_windows']!=0:
            raise ValueError('分支只支持完整epoch边界恢复。')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng']); state=saved['state']
        copy_atomic(out/f'checkpoints/epoch{state["best_epoch"]:03d}.pt',out/'checkpoints/best.pt')
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/'config.json',identity)
        state=dict(branch_epoch=0,updates=0,best_CE2=None,best_epoch=None,early_stop_counter=0)
        initial=dict(model=tensor_tree_hash(model.state_dict()),optimizer=tensor_tree_hash(optimizer.state_dict()),
            rng=tensor_tree_hash(rng_state()), first_order_sha256=order_hash(dataset.epoch_order(139)),
            restoration='model + AdamW moments/steps + RNG; LR intentionally reset to fixed 1e-5; original cosine not replayed',
            parent_epoch=138,parent_optimizer_steps=sorted({int(s['step']) for s in optimizer.state.values()}))
        write_json(out/'initial_identity.json',initial)
    reader=WindowReader(dataset,args.device)
    small={s:dataset.pairs(s)[:2] for s in ('train','full')}
    if args.mode=='check':
        result=evaluate_epoch(model,reader,out/'evaluation',dict(sha256=config['parent_sha256'],epoch=138),identity,small)
        write_json(out/'check_result.json',dict(status='passed',optimizer_updates=0,metrics=result))
        return
    budget=2 if args.mode=='smoke' else config['epochs']
    def finish_evaluation():
        epoch=state['branch_epoch']; p=out/f'checkpoints/epoch{epoch:03d}.pt'
        rows=state['group_rows']
        write_csv(out/f'updates/epoch{epoch:03d}.csv',rows,list(rows[0]))
        result=evaluate_epoch(model,reader,out/f'evaluation/epoch{epoch:03d}',
            dict(sha256=file_hash(p),branch_epoch=epoch,parent_epoch=138),identity,
            small if args.mode=='smoke' else None)
        record=dict(branch_epoch=epoch,source_order_epoch=138+epoch,online=state['online'],
            fixed_train=result['train'],fixed_validation=result['full'],gradient=state['gradient'],
            best_epoch=state['best_epoch'],is_best=state['best_epoch']==epoch,lr=1e-5,
            order_sha256=state['order_sha256'],updates=state['updates'])
        write_json(out/f'epochs/epoch{epoch:03d}.json',record)
    if state['branch_epoch']>0:
        finish_evaluation()  # 训练已保存但评价中断时，仅补该固定评价，不重放训练。
    for epoch in range(state['branch_epoch']+1,budget+1):
        order=dataset.epoch_order(138+epoch)
        if args.mode=='smoke': order=order[:7]
        mean=MetricMean(); objectives=[]; norms=[]; group_rows=[]; start=time.perf_counter()
        for offset in range(0,len(order),4):
            pairs=order[offset:offset+4]
            m,objs,norm=train_group(model,reader,pairs,optimizer,beta)
            for k in mean.sums: mean.sums[k]+=m.sums[k]
            mean.denominator+=m.denominator;mean.windows+=m.windows
            objectives.extend(objs); norms.append(norm);state['updates']+=1
            group_rows.append(dict(update=state['updates'],cursor=offset+len(pairs),windows=len(pairs),
                grad_norm_pre_clip=norm,clipped=norm>1.,lr=1e-5))
            if len(norms)%500==0: print(f'branch {epoch}: {offset+len(pairs)}/{len(order)}',flush=True)
        online=grouped(mean.result());online['L_beta']=float(np.mean(objectives));online['windows']=len(order)
        is_best=state['best_CE2'] is None or online['CE2']<state['best_CE2']
        if is_best: state['best_CE2']=online['CE2'];state['best_epoch']=epoch;state['early_stop_counter']=0
        else: state['early_stop_counter']+=1
        state.update(branch_epoch=epoch,online=online,order_sha256=order_hash(order),group_rows=group_rows,gradient=dict(
            pre_clip_mean=float(np.mean(norms)),pre_clip_max=max(norms),clip_fraction=float(np.mean(np.array(norms)>1)),
            optimizer_steps=len(norms),tail_windows=len(order)%4 or 4,seconds=time.perf_counter()-start))
        checkpoint=out/f'checkpoints/epoch{epoch:03d}.pt'
        save_checkpoint(checkpoint,model,optimizer,state,identity)
        copy_atomic(checkpoint,out/'checkpoints/last.pt')
        if is_best: copy_atomic(checkpoint,out/'checkpoints/best.pt')
        write_csv(out/f'updates/epoch{epoch:03d}.csv',group_rows,list(group_rows[0]))
        finish_evaluation()
        print(f'branch epoch {epoch}: online original CE2={online["CE2"]:.9g}, L_beta={online["L_beta"]:.9g}',flush=True)
        if args.stop_after_epoch==epoch: break
    records=[read_json(p) for p in sorted((out/'epochs').glob('epoch*.json'))]
    final=dict(completed_epochs=state['branch_epoch'],planned_epochs=budget,best_epoch=state['best_epoch'],
        best_selection='online original train CE2 only',primary_comparison='common branch epoch10; no posthoc best substitution',
        endpoint=records[-1],last3={})
    for scope in ('online','fixed_train','fixed_validation'):
        final['last3'][scope]={}
        for key in ('CE2','Q_ref','C_mean','P_mean','V_MSE','current_relative_MSE'):
            values=[r[scope][key] for r in records[-3:]]
            final['last3'][scope][key]=dict(n=len(values),mean=float(np.mean(values)),std=float(np.std(values)),
                min=min(values),max=max(values))
    write_json(out/'summary.json',final)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('check','smoke','train'))
    p.add_argument('--config',required=True);p.add_argument('--dataset-root',required=True)
    p.add_argument('--parent-checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-epoch',type=int)
    run(p.parse_args())


if __name__=='__main__': main()
