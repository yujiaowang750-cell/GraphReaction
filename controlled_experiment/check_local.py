"""隔离CPU真实窗口检查：原目标、4+3尾组、完整恢复、配对初态。"""

import argparse
import copy
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controlled_experiment.common import (load_parent,initialize_branch,objective,contributions,
    tensor_tree_hash,read_json,write_json)
from controlled_experiment.run import train_group,save_checkpoint
from full_training_windows import FullTrainingWindows
from training_windows import WindowReader
from training_step import train_group as original_group
from training_checkpoint import restore_rng,rng_state
from train import configure_runtime
from loss import compute_loss


def compare_models(a,b):
    maximum=0.
    for k,x in a.state_dict().items():
        y=b.state_dict()[k]
        torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-4)
        maximum=max(maximum,float((x-y).abs().max()))
    return maximum


def compare_optimizer(a,b):
    maximum=0.
    for key,x in a.state_dict()['state'].items():
        for name,v in x.items():
            w=b.state_dict()['state'][key][name]
            torch.testing.assert_close(v,w,atol=2e-5,rtol=2e-4)
            maximum=max(maximum,float((v-w).abs().max()))
    return maximum


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',required=True);p.add_argument('--parent-checkpoint',required=True)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    configure_runtime(torch.device('cpu'))
    d=FullTrainingWindows(args.dataset_root);c=read_json(args.config)
    parent=load_parent(args.parent_checkpoint,d,c['parent_sha256'])
    parent_optimizer_hash=tensor_tree_hash(parent['optimizer'])
    a,oa=initialize_branch(parent,'cpu');initial_a=(tensor_tree_hash(a.state_dict()),tensor_tree_hash(oa.state_dict()),tensor_tree_hash(rng_state()))
    b,ob=initialize_branch(parent,'cpu');initial_b=(tensor_tree_hash(b.state_dict()),tensor_tree_hash(ob.state_dict()),tensor_tree_hash(rng_state()))
    assert initial_a==initial_b
    assert all(oa.state[x]['exp_avg'].data_ptr()!=ob.state[y]['exp_avg'].data_ptr()
               for x,y in zip(a.parameters(),b.parameters()))
    reader=WindowReader(d,'cpu');pairs=d.epoch_order(139)[:7]
    data,current,future,target=reader.window('train',*pairs[0]);losses=compute_loss(a(data)['prediction'],target,current,data)
    assert objective(losses,1.) is losses['loss']
    explicit=sum(contributions(losses).values())
    formula_error=float((explicit-losses['loss']).abs().detach())
    torch.testing.assert_close(explicit,losses['loss'],atol=1e-9,rtol=2e-4)
    beta_formula_error=float((objective(losses,1.25)-(explicit+.25*contributions(losses)['P'])).abs().detach())
    del losses,explicit
    differences=[];metric_errors=[];norm_errors=[]
    for group in (pairs[:4],pairs[4:]):
        original,norm_a=original_group(a,reader,group,oa)
        new,objectives,norm_b=train_group(b,reader,group,ob,1.)
        assert original.windows==new.windows==len(group)
        for key,value in original.result().items():
            other=new.result()[key]
            torch.testing.assert_close(torch.tensor(value,dtype=torch.float64),torch.tensor(other,dtype=torch.float64),atol=1e-9,rtol=2e-4)
            metric_errors.append(abs(value-other))
        differences.append(compare_models(a,b))
        torch.testing.assert_close(torch.tensor(norm_a),torch.tensor(norm_b),atol=2e-5,rtol=2e-4)
        norm_errors.append(abs(norm_a-norm_b))
    optimizer_error=compare_optimizer(oa,ob)
    assert {int(s['step']) for s in ob.state.values()}=={791018}
    # 验证尾组3是3窗口均值，而非除4：与一次合并损失的独立参考比较。
    ref,oref=initialize_branch(parent,'cpu');ref.train();oref.zero_grad(set_to_none=True)
    total=0.
    for pair in pairs[4:]:
        data,current,future,target=reader.window('train',*pair)
        total=total+compute_loss(ref(data)['prediction'],target,current,data)['loss']/3
    total.backward();torch.nn.utils.clip_grad_norm_(ref.parameters(),1.,error_if_nonfinite=True);oref.step()
    tail,otail=initialize_branch(parent,'cpu');train_group(tail,reader,pairs[4:],otail,1.)
    tail_error=compare_models(ref,tail)
    # 保存后故意消耗随机数，再完整恢复，比较后续同一组更新。
    saved_path=out/'isolated.pt';save_checkpoint(saved_path,b,ob,{'branch_epoch':0},{'check':'isolated'})
    baseline_rng=rng_state(); train_group(b,reader,pairs[:4],ob,1.)
    torch.rand(13)
    saved=torch.load(saved_path,map_location='cpu',weights_only=True)
    restored,orestored=initialize_branch(parent,'cpu')
    restored.load_state_dict(saved['model'],strict=True);orestored.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    assert tensor_tree_hash(baseline_rng)==tensor_tree_hash(rng_state())
    train_group(restored,reader,pairs[:4],orestored,1.)
    resume_error=compare_models(b,restored)
    resume_optimizer_error=compare_optimizer(ob,orestored)
    # B候选也执行一组真实更新，检查有限性；没有正式训练。
    candidate,oc=initialize_branch(parent,'cpu');_,objs,norm=train_group(candidate,reader,pairs[4:],oc,1.25)
    assert all(torch.isfinite(x).all() for x in candidate.state_dict().values())
    assert tensor_tree_hash(parent['optimizer'])==parent_optimizer_hash
    write_json(out/'checks.json',dict(status='passed',device='cpu',initial_A_B_identical=True,
        beta1_original_tensor=True,formula_absolute_error=formula_error,beta1p25_formula_error=beta_formula_error,
        original_vs_new_model_max_errors=differences,tail3_mean_reference_max_error=tail_error,
        optimizer_max_error=optimizer_error,resume_model_max_error=resume_error,
        resume_optimizer_max_error=resume_optimizer_error,metric_max_error=max(metric_errors),norm_max_error=max(norm_errors),
        value_tolerance=[1e-9,2e-4],gradient_tolerance=[2e-5,2e-4],
        candidate_pre_clip_norm=norm,actual_group_sizes=[4,3],parent_unchanged=True,
        original_experiment_optimizer_updates=0,isolated_optimizer_updates=9,test_dynamic_reads=0))
    print('isolated checks PASS',flush=True)


if __name__=='__main__':main()
