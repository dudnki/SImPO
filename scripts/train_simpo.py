"""
train_simpo.py
==============
SimPO 학습 엔트리포인트 (Hydra 설정 기반)

사용법:
    python scripts/train_simpo.py
    python scripts/train_simpo.py simpo.beta=3.0 simpo.gamma=0.8
    python scripts/train_simpo.py model.name=unsloth/Llama-3.2-1B-Instruct-bnb-4bit
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig

from src.training.simpo import SimPORunConfig, train


@hydra.main(config_path="../configs", config_name="train_simpo", version_base=None)
def main(cfg: DictConfig):
    run_cfg = SimPORunConfig(
        # 모델
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        lora_target_modules=list(cfg.model.lora_target_modules),
        # 데이터
        data_path=cfg.data.path,
        max_length=cfg.data.max_length,
        max_prompt_length=cfg.data.max_prompt_length,
        # SimPO
        beta=cfg.simpo.beta,
        gamma=cfg.simpo.gamma,
        # 학습
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        max_grad_norm=cfg.training.max_grad_norm,
        optim=cfg.training.optim,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        # MLflow
        mlflow_tracking_uri=cfg.mlflow.tracking_uri,
        mlflow_experiment_name=cfg.mlflow.experiment_name,
        # Orthogonal Loss
        tau_prev_path=cfg.orthogonal.tau_prev_path,
        orthogonal_alpha=cfg.orthogonal.alpha,
    )

    train(run_cfg)


if __name__ == "__main__":
    main()
