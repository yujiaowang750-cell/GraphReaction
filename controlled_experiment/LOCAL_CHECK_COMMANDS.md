# 本轮本机检查命令

以下在 `NewProgramByMe` 根目录、原 `.venv` 内执行。输出必须为新的空目录；不要把这些命令指向旧实验目录。目录版本后缀用于保留早期检查失败现场。

```powershell
$py = '.\.venv\Scripts\python.exe'
$checkpoint138 = 'GraphReaction\training_runs\full_train_ce2_20261026_215337\checkpoints\best.pt'
$checkpoint155 = 'GraphReaction\training_runs\full_continue_193_fixed_lr_20261030_101435\checkpoints\last.pt'
$out = 'GraphReaction\training_runs\local_audit_20261031'

& $py -u GraphReaction\audit_tools\data_audit.py --dataset-root Datasets --runs GraphReaction\training_runs --output "$out\data_history_v3"
& $py -u GraphReaction\audit_tools\change_analysis.py --dataset-root Datasets --trend GraphReaction\training_runs\checkpoint_trend_20261031 --output "$out\changes_v2"
& $py -u GraphReaction\audit_tools\gradient_audit.py --dataset-root Datasets --checkpoint $checkpoint138 --sha256 bddb304aaa19de2b8dc0c00acc438cc91b45741049c584b3dfa98f08d1b9d689 --checkpoint155 $checkpoint155 --sha256155 85d5b838613ac3fa7e4c72841efce26e5fd8b58cb5a333de4be46fc5fc1aa758 --output "$out\gradient_v2" --device cuda
& $py -u GraphReaction\controlled_experiment\check_local.py --dataset-root Datasets --parent-checkpoint $checkpoint138 --config GraphReaction\controlled_experiment\configs\A_beta1.json --output "$out\isolated_checks_v3"

& $py -u GraphReaction\controlled_experiment\run.py smoke --config GraphReaction\controlled_experiment\configs\A_beta1.json --dataset-root Datasets --parent-checkpoint $checkpoint138 --output "$out\smoke_A_v2" --stop-after-epoch 1 --device cuda
& $py -u GraphReaction\controlled_experiment\run.py smoke --config GraphReaction\controlled_experiment\configs\A_beta1.json --dataset-root Datasets --parent-checkpoint $checkpoint138 --output "$out\smoke_A_v2" --resume --device cuda
& $py -u GraphReaction\controlled_experiment\run.py check --config GraphReaction\controlled_experiment\configs\B_beta1p25.json --dataset-root Datasets --parent-checkpoint $checkpoint138 --output "$out\check_B" --device cuda
& $py -u GraphReaction\controlled_experiment\run.py smoke --config GraphReaction\controlled_experiment\configs\B_beta1p25.json --dataset-root Datasets --parent-checkpoint $checkpoint138 --output "$out\smoke_B" --device cuda
```

成功结果与数值误差见审计报告。四个原checkpoint另做strict加载及参数哈希读回；对原保护清单与源文件做了末尾SHA256复核。

`compare.py --help`、新Python文件AST解析已运行；正式10轮结果的 `compare.py` 尚无输入，未运行其完整绘图/对比。没有用虚构的十轮数据充当试验结果。

这些命令不需要云端。云端路径、短检查、正式启动及恢复命令另见README。
