"""复用154/155已有完整评价，区分字段、时间箱、区域和端电压偏移。"""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controlled_experiment.common import read_json,write_json,grouped
from run_files import write_csv,file_hash
from full_training_windows import FullTrainingWindows


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',required=True);p.add_argument('--trend',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();trend=Path(args.trend);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    dataset=FullTrainingWindows(args.dataset_root)
    checks=read_json(trend/'evaluation_checks.json')['groups']
    coverage=[];raw={};max_formula_error=0.
    for key,entry in checks.items():
        folder=Path(entry['source']);epoch,split=key.split('_');scope='train' if split=='train' else 'full'
        split='train' if scope=='train' else 'validation'
        overall=read_json(folder/'overall_metrics.json');trajectories=read_json(folder/'trajectory_metrics.json')
        vc=read_csv(folder/'voltage_current.csv');pairs=[(int(r['sample_index']),int(r['input_time_index'])) for r in vc]
        assert len(pairs)==len(set(pairs)) and set(pairs)==set(dataset.pairs(scope))
        assert all(int(r['target_time_index'])==int(r['input_time_index'])+1 and float(r['weight'])==1 for r in vc)
        assert overall['identity']['data']==dataset.identity
        assert overall['checkpoint']['sha256']==entry['checkpoint']['sha256']
        counts={sid:sum(s==sid for s,t in pairs) for sid in set(s for s,t in pairs)}
        losses=overall['losses'];fromtraj={k:sum(counts[int(sid)]*v[k] for sid,v in trajectories.items())/len(pairs) for k in losses}
        # RMSE必须在整体MSE上开方，不能平均各轨迹RMSE。
        fromtraj['current_relative_RMSE_pct']=100*np.sqrt(fromtraj['current_relative_MSE'])
        for k in losses: max_formula_error=max(max_formula_error,abs(losses[k]-fromtraj[k]))
        g=grouped(losses)
        max_formula_error=max(max_formula_error,abs(losses['CE2']-sum(g[k+'_contribution'] for k in 'CPVI')))
        max_formula_error=max(max_formula_error,abs(losses['Q_ref']-(sum(losses[n+'_MSE'] for n in ('ce','phi_e','cg','cp','phi_p','phi_CBD','V'))/7+.1*losses['current_relative_MSE'])))
        for field in ('ce','phi_e'):
            max_formula_error=max(max_formula_error,abs(losses[field+'_MSE']-.5*(losses[field+'_phase2_MSE']+losses[field+'_phase5_MSE'])))
        coverage.append(dict(epoch=int(epoch),split=split,windows=len(pairs),trajectories=len(counts),
            overall_sha256=file_hash(folder/'overall_metrics.json'),source=str(folder),checkpoint=entry['checkpoint']))
        raw[(int(epoch),split)]=dict(folder=folder,overall=overall,trajectories=trajectories,vc=vc,counts=counts)
    assert max_formula_error<1e-12
    write_json(out/'evaluation_coverage.json',dict(groups=coverage,max_aggregation_error=max_formula_error,
        window_average=True,trajectory_weight='number of windows',node_average='separate physical_node_pooled metrics',test_dynamic_reads=0))
    differences=[];trajectory_rows=[];summary={}
    for split in ('train','validation'):
        a,b=raw[(154,split)],raw[(155,split)]
        for name,keynames in [('field_metrics.csv',('field','phase')),('stage_metrics.csv',('stage','field')),('region_metrics.csv',('region_or_interface','field'))]:
            old={tuple(r[k] for k in keynames):r for r in read_csv(a['folder']/name)}
            new={tuple(r[k] for k in keynames):r for r in read_csv(b['folder']/name)}
            assert old.keys()==new.keys()
            for key,r in old.items():
                after=new[key];x=r['standardized_node_pooled_MSE'];y=after['standardized_node_pooled_MSE']
                if x=='' or y=='':continue
                x=float(x);y=float(y)
                differences.append(dict(split=split,table=name,key='|'.join(key),MSE154=x,MSE155=y,
                    change=y-x,relative_change_pct=100*(y/x-1) if x else None,
                    count154=int(r['count']),count155=int(after['count'])))
        for sid,old in a['trajectories'].items():
            new=b['trajectories'][sid];old=grouped(old);new=grouped(new)
            trajectory_rows.append(dict(split=split,sid=int(sid),trajectory=dataset.samples[int(sid)]['trajectory_id'],windows=a['counts'][int(sid)],
                **{f'{name}_{epoch}':v[name] for epoch,v in ((154,old),(155,new)) for name in ('CE2','C_mean','P_mean','V_MSE','current_relative_MSE')},
                P_weighted_change=(new['P_mean']-old['P_mean'])*a['counts'][int(sid)]/len(a['vc'])))
        summary[split]={}
        for epoch,block in ((154,a),(155,b)):
            errors=np.array([float(r['voltage_error_mV']) for r in block['vc']])
            summary[split][str(epoch)]=dict(voltage_signed_mean_mV=float(errors.mean()),
                voltage_error_std_mV=float(errors.std()),voltage_RMSE_mV=float(np.sqrt(np.mean(errors**2))),
                bias_squared_fraction_of_MSE=float(errors.mean()**2/np.mean(errors**2)))
        selected=[r for r in trajectory_rows if r['split']==split]
        signed=np.array([r['P_weighted_change'] for r in selected]);improvements=np.sort(-signed[signed<0])[::-1]
        summary[split]['potential']=dict(improved=sum(r['P_mean_155']<r['P_mean_154'] for r in selected),total=len(selected),
            top3_fraction_of_gross_improvement=float(improvements[:3].sum()/improvements.sum()),
            gross_improvement=float(improvements.sum()),net_change=float(signed.sum()))
    write_csv(out/'field_stage_region_changes.csv',differences,list(differences[0]))
    write_csv(out/'trajectory_changes.csv',trajectory_rows,list(trajectory_rows[0]))
    write_json(out/'change_summary.json',summary)
    print('saved evaluation coverage and change analysis',flush=True)


if __name__=='__main__':main()
