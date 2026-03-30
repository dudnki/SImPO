"""
SimPO 학습 로직
Unsloth + TRL CPOTrainer (loss_type="simpo") 기반
OrthogonalSimPOTrainer: Orthogonal Loss로 태스크 간 간섭 최소화
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import mlflow
import torch
from datasets import load_from_disk
from trl import CPOConfig, CPOTrainer
from unsloth import FastLanguageModel


@dataclass
class SimPORunConfig:
    # 모델
    model_name: str
    max_seq_length: int
    load_in_4bit: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    # 데이터
    data_path: str
    max_length: int
    max_prompt_length: int
    # SimPO
    beta: float
    gamma: float
    # 학습
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    max_grad_norm: float
    optim: str
    bf16: bool
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    # MLflow
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    # Orthogonal Loss
    tau_prev_path: Optional[str] = None
    orthogonal_alpha: float = 0.0


class OrthogonalSimPOTrainer(CPOTrainer):
    """
    CPOTrainer + Orthogonal Loss
    L_total = L_SimPO + alpha * cos²(Δθ_i, Δθ_j)

    cos²은 trace trick으로 O(r²) 연산 (r=LoRA rank)
    """

    def __init__(self, *args, tau_prev_path: str = None, alpha: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.tau_prev = None

        if tau_prev_path and alpha > 0.0:
            raw = torch.load(tau_prev_path, map_location="cpu", weights_only=True)
            self.tau_prev = raw
            print(f"[OrthogonalSimPO] tau_prev 로드: {len(raw)} 레이어, alpha={alpha}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        if self.alpha > 0.0 and self.tau_prev is not None:
            if return_outputs:
                loss, outputs = result
            else:
                loss = result

            orth_loss = self._orthogonal_loss(model)
            loss = loss + self.alpha * orth_loss

            if self.state.global_step % self.args.logging_steps == 0:
                self.log({"orthogonal_loss": orth_loss.item()})

            return (loss, outputs) if return_outputs else loss

        return result

    def _orthogonal_loss(self, model) -> torch.Tensor:
        """
        Trace trick: cos²(ΔWi, ΔWj) in O(r²)
        ⟨ΔWi, ΔWj⟩_F = tr(Bi^T Bj · Aj Ai^T)
        """
        device = next(model.parameters()).device
        total_cos2 = torch.tensor(0.0, device=device)
        count = 0

        for name, module in model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if name not in self.tau_prev:
                continue

            adapter_key = list(module.lora_A.keys())[0]
            A_i = module.lora_A[adapter_key].weight  # (r, in)
            B_i = module.lora_B[adapter_key].weight  # (out, r)

            A_j = self.tau_prev[name]["A"].to(device=device, dtype=A_i.dtype)
            B_j = self.tau_prev[name]["B"].to(device=device, dtype=B_i.dtype)

            # Trace trick
            C = B_i.T @ B_j           # (r, r)
            D = A_j @ A_i.T           # (r, r)
            inner = torch.trace(C @ D)

            norm_i2 = torch.trace(B_i.T @ B_i @ A_i @ A_i.T)
            norm_j2 = torch.trace(B_j.T @ B_j @ A_j @ A_j.T)

            cos2 = inner ** 2 / (norm_i2 * norm_j2 + 1e-8)
            total_cos2 = total_cos2 + cos2
            count += 1

        return total_cos2 / max(count, 1)


def load_model_and_tokenizer(cfg: SimPORunConfig):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.lora_target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer.padding_side = "left"  # CPOTrainer 요구사항
    return model, tokenizer


def load_datasets(cfg: SimPORunConfig):
    dataset = load_from_disk(cfg.data_path)
    return dataset["train"], dataset["test"]


def build_trainer(model, tokenizer, train_ds, eval_ds, cfg: SimPORunConfig) -> CPOTrainer:
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    cpo_config = CPOConfig(
        output_dir=cfg.output_dir,
        loss_type="simpo",
        beta=cfg.beta,
        simpo_gamma=cfg.gamma,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim,
        bf16=cfg.bf16,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        logging_steps=cfg.logging_steps,
        report_to="mlflow",
    )

    use_orthogonal = cfg.orthogonal_alpha > 0.0 and cfg.tau_prev_path is not None
    TrainerClass = OrthogonalSimPOTrainer if use_orthogonal else CPOTrainer

    kwargs = dict(
        model=model,
        args=cpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    if use_orthogonal:
        kwargs["tau_prev_path"] = cfg.tau_prev_path
        kwargs["alpha"] = cfg.orthogonal_alpha

    return TrainerClass(**kwargs)


def train(cfg: SimPORunConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    with mlflow.start_run(run_name="train_simpo"):
        mlflow.log_params({
            "model": cfg.model_name,
            "lora_r": cfg.lora_r,
            "beta": cfg.beta,
            "gamma": cfg.gamma,
            "lr": cfg.learning_rate,
            "batch_size": cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
            "epochs": cfg.num_train_epochs,
            "data_path": cfg.data_path,
            "orthogonal_alpha": cfg.orthogonal_alpha,
            "tau_prev_path": cfg.tau_prev_path or "none",
        })

        print("모델 로드 중...")
        model, tokenizer = load_model_and_tokenizer(cfg)

        print("데이터셋 로드 중...")
        train_ds, eval_ds = load_datasets(cfg)
        mlflow.log_metric("train_samples", len(train_ds))
        mlflow.log_metric("eval_samples", len(eval_ds))

        print(f"학습 시작: train={len(train_ds)}, eval={len(eval_ds)}")
        trainer = build_trainer(model, tokenizer, train_ds, eval_ds, cfg)
        trainer.train()

        print(f"어댑터 저장: {cfg.output_dir}")
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        mlflow.log_param("adapter_path", cfg.output_dir)

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            mlflow.log_metric("peak_vram_gb", round(vram_gb, 2))
            print(f"피크 VRAM: {vram_gb:.2f} GB")

    print("학습 완료")
