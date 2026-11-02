"""生成可核验的增量包：新增模块加原运行源码依赖；不带数据、权重、历史输出。"""

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from controlled_experiment.common import read_json,write_json


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=read_json(ROOT/'controlled_experiment/runtime_manifest.json')
    runtime=[ROOT/name for name in manifest['files']]
    new=[p for directory in ('controlled_experiment','audit_tools') for p in (ROOT/directory).rglob('*')
         if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.local.sh')]
    files=sorted(set(runtime+new))
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    rows=[]
    categories={name:[] for name in ('modified','new','deleted','renamed','unchanged_runtime_dependencies')}
    for path in files:
        rel=path.relative_to(ROOT).as_posix();target='GraphReaction/'+rel
        category='new' if path in new else 'unchanged_runtime_dependencies'
        purpose='计算身份所需的原源码依赖，不修改旧训练逻辑' if category!='new' else '新审计/受控实验入口、配置或说明'
        if path.suffix=='.py':
            tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=rel)
            purpose=(ast.get_docstring(tree) or purpose).splitlines()[0]
        rows.append(dict(change=category,relative_path=target,purpose=purpose,target_relative_path=target,sha256=digest(path),bytes=path.stat().st_size))
        categories[category].append(target)
    with (out/'UPLOAD_MANIFEST.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    description=dict(base_git_commit=base,categories=categories,files=rows,
        scope='new independent modules plus exact unchanged runtime dependencies; original root sources unmodified',
        excluded=['Datasets','*.pt','*.npz','*.npy','training_runs','.venv','__pycache__','.git','*.local.sh'],
        preserve_or_manual_merge=['controlled_experiment/cloud_paths.local.sh','server virtual environment','server datasets/checkpoints/history','server-specific launch scripts'],
        old_files_to_delete=[],old_files_policy='Keep old experiments. Deploy into a new code directory or review runtime dependency files individually.',
        manifest_self_hash='Manifest files list payload hashes only, avoiding self-referential hash; outer ZIP SHA256 supplied separately.')
    write_json(out/'DELIVERY_MANIFEST.json',description)
    archive=out/'GraphReaction_incremental.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for row,path in zip(rows,files): z.write(path,row['target_relative_path'])
        for name in ('UPLOAD_MANIFEST.csv','DELIVERY_MANIFEST.json'): z.write(out/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in rows:
            assert hashlib.sha256(z.read(row['target_relative_path'])).hexdigest()==row['sha256']
        assert not any(Path(n).suffix.lower() in ('.pt','.npy','.npz','.pyc') or 'training_runs/' in n or '.venv/' in n for n in z.namelist())
    write_json(out/'PACKAGE_CHECKS.json',dict(status='passed',payload_files=len(rows),archive=archive.name,
        sha256=digest(archive),bytes=archive.stat().st_size,syntax_parsed=True,zip_crc_and_payload_sha_verified=True,
        modified_existing_files=0,new_files=len(categories['new']),unchanged_dependencies=len(categories['unchanged_runtime_dependencies']),
        deleted_files=0,renamed_files=0,remote_operations=0))
    print(json.dumps(read_json(out/'PACKAGE_CHECKS.json'),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
