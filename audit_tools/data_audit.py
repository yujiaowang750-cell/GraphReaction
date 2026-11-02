"""只读数据/历史审计：核算训练统计，不保存或替换归一化统计。"""

import argparse
import csv
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controlled_experiment.common import read_json, write_json, tensor_tree_hash
from full_training_windows import FullTrainingWindows
from training_windows import order_hash
from run_files import file_hash, write_csv
from initialization import create_model


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


class Moments:
    def __init__(self, center):
        self.center=center; self.n=0; self.sum=0.; self.square=0.
    def add(self, values):
        values=np.asarray(values,dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("审计遇到非有限有效字段。")
        d=values-self.center
        self.n+=d.size; self.sum+=float(d.sum()); self.square+=float(np.square(d).sum())
    def result(self):
        mean=self.center+self.sum/self.n
        std=math.sqrt(max(0.,self.square/self.n-(self.sum/self.n)**2))
        return self.n,mean,std


def audit_data(dataset,out):
    root=dataset.root
    manifest=read_json(root/'dataset_manifest.json');split=read_json(root/'trajectory_split.json')
    samples=manifest['samples']
    for key in ('sample_index','sample_id','trajectory_id','processed_relative_dir','source_state_history_path'):
        assert len({s[key] for s in samples})==len(samples), key
    metadata=[]
    for sample in samples:
        local=read_json(root/sample['processed_relative_dir']/'sample_manifest.json')
        for key in ('trajectory_id','sample_id','split','input_time_indices','sample_dt_s','prediction_horizon_s'):
            assert local[key]==sample[key], (sample['sample_id'],key)
        assert sample['sample_dt_s']==sample['prediction_horizon_s']==10.
        assert sample['prediction_horizon_intervals']==1
        metadata.append(dict(sid=sample['sample_index'],trajectory=sample['trajectory_id'],split=sample['split'],
            source=sample['source_state_history_path'], source_sha256=sample.get('source_sha256'),
            windows=len(sample['input_time_indices'])))
    # 测试集仅比较清单/身份，不打开任何测试动态文件。
    with np.load(root/'window_index.npz') as z:
        for scope in ('train','validation','test'):
            pairs=list(zip(z[scope+'_sample_index'].tolist(),z[scope+'_input_time_index'].tolist()))
            expected={(s['sample_index'],t) for s in samples if s['split']==scope for t in s['input_time_indices']}
            assert len(pairs)==len(set(pairs))==split['window_counts'][scope] and set(pairs)==expected
    write_csv(out/'sample_identity.csv',metadata,list(metadata[0]))
    specs=[('electrolyte_concentration_mol_m3','electrolyte',0,None),
           ('electrolyte_potential_v','electrolyte',1,None),
           ('negative_graphite_lithiation_fraction','graphite',0,'negative_graphite_max_concentration_mol_m3'),
           ('negative_graphite_solid_potential_v','graphite',1,None),
           ('positive_am_lithiation_fraction','positive_am',0,'positive_am_max_concentration_mol_m3'),
           ('positive_am_solid_potential_v','positive_am',1,None),
           ('positive_cbd_solid_potential_v','positive_cbd',0,None),
           ('positive_collector_potential_v','collector',0,None)]
    moments={(name,prefix):Moments(dataset.statistics['dynamic'][name][prefix+'mean'])
             for name,_,_,_ in specs for prefix in ('input_','delta_')
             if prefix+'mean' in dataset.statistics['dynamic'][name]}
    geometry={k:Moments(v['mean']) for k,v in dataset.statistics['geometry'].items()}
    content=[]; max_dt_error=0.; mapping_checked=0
    for i,s in enumerate(dataset.samples.values(),1):
        folder=root/s['processed_relative_dir']; indices=np.array(s['input_time_indices'])
        with np.load(folder/'dynamic/time_and_global.npz') as z:
            times=z['time_s']; assert len(times)==s['num_selected_time_points']
            max_dt_error=max(max_dt_error,float(np.max(np.abs(times[indices+1]-times[indices]-10.))))
            assert np.allclose(times[indices+1]-times[indices],10.,rtol=0,atol=1e-12)
        with np.load(folder/'static_graph.npz') as g,np.load(folder/'node_mapping.npz') as mapping:
            phase=g['node_phase']; allnodes=[]
            phase_map={'electrolyte':(2,5),'graphite':(1,),'positive_am':(3,), 'positive_cbd':(4,), 'collector':(6,7)}
            for name,ps in phase_map.items():
                nodes=mapping[name+'_node_index']; allnodes.extend(nodes.tolist())
                assert np.array_equal(np.sort(nodes),np.where(np.isin(phase,ps))[0])
            assert np.array_equal(sorted(allnodes),np.arange(len(phase)))
            source=mapping['source_node_index']; assert len(set(source))==len(phase)
            collector_positive=np.where(phase[mapping['collector_node_index']]==7)[0]
            assert len(collector_positive)==1
            if s['split']=='train':
                geo={'node_volume_um3':g['node_volume_um3'],'throat_area_um2':g['throat_area_um2'],
                     'conduit_endpoint_length_um':g['throat_conduit_lengths_um'][:,[0,2]],
                     'conduit_throat_length_um':g['throat_conduit_lengths_um'][:,1],
                     'sample_coordinate_range_zyx_um':g['sample_coordinate_range_zyx_um']}
                for name,v in geo.items(): geometry[name].add(v)
        mapping_checked+=1
        digest=hashlib.sha256()
        for filename in ('static_graph.npz','node_mapping.npz'):
            digest.update(file_hash(folder/filename).encode())
        arrays={}
        for name in phase_map:
            path=folder/'dynamic'/f'{name}_state.npy'
            digest.update(file_hash(path).encode())
            arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
            assert arrays[name].shape[0]==len(times)
        content.append(dict(sid=s['sample_index'],split=s['split'],processed_content_sha256=digest.hexdigest()))
        for field,array_name,col,cmax in specs:
            array=arrays[array_name]
            if array_name=='collector': array=array[:,collector_positive,:]
            scale=s[cmax] if cmax else 1.
            assert np.isfinite(scale) and scale>0
            for start in range(0,len(indices),32):
                idx=indices[start:start+32]
                current=array[idx,:,col]/scale; future=array[idx+1,:,col]/scale
                assert np.isfinite(current).all() and np.isfinite(future).all(),(s['sample_id'],field,start)
                if s['split']=='train':
                    if (field,'input_') in moments: moments[(field,'input_')].add(current)
                    if (field,'delta_') in moments: moments[(field,'delta_')].add(future-current)
        if i%10==0: print(f'data audit {i}/79',flush=True)
    comparison=[]
    for (field,prefix),m in moments.items():
        ref=dataset.statistics['dynamic'][field]; n,mean,std=m.result()
        comparison.append(dict(field=field,kind=prefix,count=n,saved_count=ref[prefix+'count'],
            mean=mean,saved_mean=ref[prefix+'mean'],std=std,saved_std=ref[prefix+'raw_std'],
            mean_abs_error=abs(mean-ref[prefix+'mean']),std_abs_error=abs(std-ref[prefix+'raw_std'])))
    for field,m in geometry.items():
        ref=dataset.statistics['geometry'][field];n,mean,std=m.result()
        comparison.append(dict(field=field,kind='geometry',count=n,saved_count=ref['count'],mean=mean,
            saved_mean=ref['mean'],std=std,saved_std=ref['raw_std'],mean_abs_error=abs(mean-ref['mean']),std_abs_error=abs(std-ref['raw_std'])))
    write_csv(out/'normalization_recalculation.csv',comparison,list(comparison[0]))
    write_csv(out/'processed_content_identity.csv',content,list(content[0]))
    duplicate=[k for k,v in Counter(r['processed_content_sha256'] for r in content).items() if v>1]
    write_json(out/'data_audit.json',dict(identity=dataset.identity,split=split['trajectory_counts'],windows=split['window_counts'],
        mapping_checked=mapping_checked,time_max_abs_error=max_dt_error,all_train_val_effective_fields_finite=True,
        train_statistics_counts_match=all(r['count']==r['saved_count'] for r in comparison),
        statistics_comparison= comparison, duplicate_processed_train_val_content=duplicate,
        raw_source_hashes_missing=sum(s.get('source_sha256') is None for s in samples),
        raw_source_unavailable=manifest['source_root'], test_dynamic_reads=0,
        limits='Unique trajectory IDs and processed train/validation bytes do not prove independent generative seeds or exclude near-duplicates. Test dynamics not inspected.'))


def audit_history(dataset,runs,out):
    roots=[runs/'full_train_ce2_20261026_215337', runs/'full_continue_193_fixed_lr_20261030_101435']
    records=[]; protected={}
    for root in roots:
        for name in ('config.json','history.csv','history.json','update_metrics.csv','active_training.json','summary.json'):
            if (root/name).exists(): protected[str(root/name)]=file_hash(root/name)
        for filename in ('best.pt','last.pt'):
            path=root/'checkpoints'/filename; protected[str(path)]=file_hash(path)
            p=torch.load(path,map_location='cpu',weights_only=True)
            model=create_model(); result=model.load_state_dict(p['model'],strict=True)
            assert not result.missing_keys and not result.unexpected_keys
            assert tensor_tree_hash(model.state_dict())==tensor_tree_hash(p['model'])
            state=p['state']; opt=p['optimizer']
            assert p['identity']['data']==dataset.identity
            assert state['updates']==state['epoch']*5732 and state['cursor']==0
            steps=[int(s['step']) for s in opt['state'].values()]
            assert all(torch.isfinite(s[k]).all() for s in opt['state'].values() for k in ('exp_avg','exp_avg_sq'))
            histories=state['history']; best=min(histories,key=lambda r:r['online_train_CE2'])
            assert state['best_epoch']==best['epoch']
            snapshots=root/'source_snapshot'
            records.append(dict(path=str(path),sha256=protected[str(path)],epoch=state['epoch'],updates=state['updates'],
                best_epoch=state['best_epoch'],model_sha256=tensor_tree_hash(p['model']),optimizer_sha256=tensor_tree_hash(opt),
                rng_sha256=tensor_tree_hash(p['rng']),optimizer_step_min=min(steps),optimizer_step_max=max(steps),
                optimizer_parameters=len(steps),model_parameters=len(list(model.parameters())),
                rng_keys=list(p['rng']),scheduler=p['scheduler'],lr=opt['param_groups'][0]['lr'],
                strict_load_missing=[],strict_load_unexpected=[],moments_finite=True,pending_windows=p['accumulation_pending_windows']))
    history=read_csv(roots[1]/'history.csv'); boundary=[]
    for epoch in (138,141,142,154,155):
        row=history[epoch-1]; assert int(row['epoch'])==epoch
        assert row['order_sha256']==order_hash(dataset.epoch_order(epoch))
        boundary.append(row)
    with (roots[1]/'update_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        counts=Counter(); tails=[]; max_norm={}; clipped=Counter()
        for number,row in enumerate(csv.DictReader(f),1):
            epoch=int(row['epoch']); update=int(row['update']);cursor=int(row['cursor'])
            assert update==number
            within=(update-1)%5732+1
            assert cursor==min(within*4,22927) and epoch==(update-1)//5732+1
            if epoch>=142: assert float(row['lr'])==1e-5
            assert row['order_sha256']==history[epoch-1]['order_sha256']
            assert math.isfinite(float(row['grad_norm_pre_clip'])) and int(row['nonfinite_grad_count'])==0
            counts[epoch]+=1
            if cursor==22927: tails.append(row)
            max_norm[epoch]=max(max_norm.get(epoch,0),float(row['grad_norm_pre_clip']))
            clipped[epoch]+=row['clipped']=='True'
        assert number==888460 and all(n==5732 for n in counts.values())
    # 原始141轮日志应为续训日志的完整前缀。
    with (roots[0]/'update_metrics.csv').open(encoding='utf-8-sig') as a,(roots[1]/'update_metrics.csv').open(encoding='utf-8-sig') as b:
        assert all(x==next(b) for x in a)
    write_json(out/'checkpoint_audit.json',records)
    write_json(out/'history_audit.json',dict(update_rows=number,steps_per_epoch=5732,tail_windows=3,
        all_epochs_have_tail_step=True,original_log_is_continuation_prefix=True,boundary_history=boundary,
        tail_steps=[r for r in tails if int(r['epoch']) in (138,141,142,154,155)],
        clipping={str(e):dict(max=max_norm[e],fraction=clipped[e]/counts[e]) for e in (138,141,142,154,155)},
        optimizer_state_continuity_limit='141 and154/155 states exist; no142-start state was saved. Resume source restores moments/RNG; cannot empirically replay missing intermediate checkpoint.',
        rng_order='Order deterministically recreated from MT19937 seed2026 and epoch, independent of global RNG.'))
    write_json(out/'protected_original_hashes.json',protected)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',required=True);p.add_argument('--runs',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    d=FullTrainingWindows(args.dataset_root)
    audit_history(d,Path(args.runs),out)
    audit_data(d,out)
    print('data and history audit completed',flush=True)


if __name__=='__main__': main()
