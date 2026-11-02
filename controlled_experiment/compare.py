"""A/B结束后画完整曲线，主要比较共同第10轮，同时列出最后3轮波动。"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controlled_experiment.common import read_json,write_json
from run_files import write_csv


def main():
    p=argparse.ArgumentParser();p.add_argument('--a',required=True);p.add_argument('--b',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    roots={'A':Path(args.a),'B':Path(args.b)}
    initial=[read_json(r/'initial_identity.json') for r in roots.values()]
    assert initial[0]==initial[1], 'A/B起点模型、优化器、RNG或顺序不一致'
    records={k:[read_json(p) for p in sorted(r.glob('epochs/epoch*.json'))] for k,r in roots.items()}
    assert all([x['branch_epoch'] for x in rows]==list(range(1,11)) for rows in records.values()), '主比较要求两组完整10轮'
    for a,b in zip(records['A'],records['B']):
        assert a['order_sha256']==b['order_sha256'] and a['updates']==b['updates']
    keys=['CE2','Q_ref','C_mean','P_mean','V_MSE','current_relative_MSE']
    keys+=['ce_MSE','cg_MSE','cp_MSE','phi_e_MSE','phi_p_MSE','phi_CBD_MSE']
    table=[]
    for branch,rows in records.items():
        for row in rows:
            for scope in ('online','fixed_train','fixed_validation'):
                table.append(dict(branch=branch,epoch=row['branch_epoch'],scope=scope,**{k:row[scope][k] for k in keys}))
    write_csv(out/'curves.csv',table,list(table[0]))
    fig,axes=plt.subplots(3,4,figsize=(17,10),layout='constrained')
    for key,ax in zip(keys,axes.flat):
        for branch,rows in records.items():
            for scope,style in [('fixed_train','--'),('fixed_validation','-')]:
                ax.plot(range(1,11),[r[scope][key] for r in rows],style,label=branch+' '+scope,linewidth=1.5)
        ax.set_title(key);ax.set_xlabel('Branch epoch');ax.set_yscale('log');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(out/'fixed_curves.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for branch,rows in records.items():
        ax.plot(range(1,11),[r['online']['CE2'] for r in rows],label=branch+' original CE2')
        ax.plot(range(1,11),[r['online']['L_beta'] for r in rows],'--',label=branch+' objective L_beta')
    ax.legend();ax.set_xlabel('Branch epoch');ax.set_ylabel('Online window mean loss');fig.savefig(out/'online_objectives.png',dpi=180);plt.close(fig)
    comparison={}
    for scope in ('fixed_train','fixed_validation'):
        comparison[scope]={}
        for key in keys:
            a,b=records['A'][-1][scope][key],records['B'][-1][scope][key]
            comparison[scope][key]=dict(A_epoch10=a,B_epoch10=b,B_minus_A=b-a,
                last3={branch:dict(mean=float(np.mean([r[scope][key] for r in rows[-3:]])),
                    std=float(np.std([r[scope][key] for r in rows[-3:]])),
                    min=min(r[scope][key] for r in rows[-3:]),max=max(r[scope][key] for r in rows[-3:])) for branch,rows in records.items()})
    write_json(out/'comparison.json',dict(primary='common endpoint epoch10',best_rule='online original CE2',metrics=comparison))


if __name__=='__main__':main()
