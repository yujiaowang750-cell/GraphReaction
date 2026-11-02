"""32档与全量训练设置；模型参数和学习率公式分别在原来的配置文件中。"""

from learning_rate import NUM_EPOCHS, INITIAL_LR, MIN_LR

SEED = 2027
ACCUMULATION = 4
CLIP_NORM = 1.0
EARLY_STOPPING = 20


def training_configuration(dataset_mode="proxy32"):
    if dataset_mode not in ("proxy32", "full"):
        raise ValueError("dataset_mode 只能是 proxy32 或 full。")
    settings = dict(
        schema="graphreaction_train_ce2_metrics_v3", seed=SEED, max_epochs=NUM_EPOCHS,
        optimizer="AdamW", learning_rate=INITIAL_LR, weight_decay=1e-6,
        betas=[0.9, 0.999], adam_eps=1e-8, accumulation=ACCUMULATION, clip_norm=CLIP_NORM,
        scheduler="CosineAnnealingLR", T_max=NUM_EPOCHS, eta_min=MIN_LR,
        scheduler_step="epoch_end_after_optimizer", monitor="online_train_CE2",
        best_rule="strict_decrease", early_stopping=EARLY_STOPPING,
        validate_each_epoch=False, diagnostic_during_training=False,
        diagnostic_interval=None, diagnostic_panel="final_best_only_first_train_and_proxy_window",
        shuffle="original two-level MT19937: trajectories then windows", static_cache_graphs=1,
        amp=False, tf32=False, workers=0, torch_threads=4,
        deterministic_algorithms=False, cudnn_deterministic=True,
        train_weight="mean_loss_weight", proxy_weight="population_weight", full_weight=1,
        reported_window_weight=1, reported_nrmse="unweighted_node_pooled_delta",
        save_every_epoch=True, epoch_checkpoint_pattern="checkpoints/epoch_{epoch:04d}.pt",
        group_metrics="C=(2ce+cg+cp)/4; P=(phi_e+phi_p+phi_CBD)/3; V; I_relative_MSE",
    )
    settings.update(dataset_mode=dataset_mode, evaluation_scopes=["train", "proxy", "full"])
    if dataset_mode == "full":
        settings.update(
            schema="graphreaction_full_dual_best_val_stop_v2",
            train_weight=1, proxy_weight=None, evaluation_scopes=["full"],
            diagnostic_panel="final_best_only_first_train_and_full_validation_window",
            validate_each_epoch=True, validation_scope="full", validation_weight=1,
            train_best_monitor="online_train_CE2", validation_best_monitor="val_CE2",
            monitor="val_CE2", early_stopping_monitor="val_CE2", min_delta=0., final_checkpoint="best_val",
        )
    return settings
