"""
4단계: Task Arithmetic 적용 + 다중 태스크 벡터 합성 평가
θ_new = θ_base + λ1*τ1 + λ2*τ2 + ...

fp16 베이스 모델에 task vector를 직접 합산하는 진짜 Task Arithmetic.
각 λ마다 base state를 복원해서 재사용하므로 모델을 다시 로드하지 않아도 됨.
"""
import logging
import sys
import torch
import mlflow
import hydra
from omegaconf import DictConfig
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.metrics import compute_simpo_margins

log = logging.getLogger(__name__)


def load_4bit_and_dequantize(base_model_path: str):
    """
    4-bit 모델을 로드하고 bf16으로 dequantize.
    학습 환경(4-bit base + LoRA)과 동일한 base 가중치 공간에서 Task Arithmetic 수행.
    """
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig

    log.info(f"4-bit 모델 로드 후 dequantize: {base_model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
    )

    # Linear4bit → 일반 nn.Linear(bf16)으로 교체
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, bnb.nn.Linear4bit):
                dequant = child.weight.dequantize().to(torch.bfloat16).cpu()
                bias = child.bias.to(torch.bfloat16).cpu() if child.bias is not None else None
                new_linear = torch.nn.Linear(
                    dequant.shape[1], dequant.shape[0],
                    bias=bias is not None, dtype=torch.bfloat16, device="cpu"
                )
                new_linear.weight = torch.nn.Parameter(dequant)
                if bias is not None:
                    new_linear.bias = torch.nn.Parameter(bias)
                setattr(module, child_name, new_linear)

    model = model.cpu()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def apply_multi_vector(model, vectors: list[dict], lambda_val: float):
    """
    θ = θ_base + λ * (τ1 + τ2 + ...)
    model을 in-place로 수정. 호출 전 base_state로 복원 필요.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            for vec in vectors:
                if name in vec:
                    param.data.add_(lambda_val * vec[name].to(param.dtype))


@hydra.main(config_path="../configs", config_name="apply_arithmetic", version_base=None)
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # 4-bit 모델 로드 후 bf16 dequantize (학습 환경과 동일한 base)
    model, tokenizer = load_4bit_and_dequantize(cfg.base_model)

    # base state 저장 (각 λ마다 복원용)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    log.info("base state 저장 완료")

    # task vector 로드
    vectors = []
    vector_names = []
    for vp in cfg.vector_paths:
        vec = torch.load(vp, map_location="cpu", weights_only=True)
        vectors.append(vec)
        vector_names.append(Path(vp).stem)
    log.info(f"로드된 벡터: {vector_names}")

    # 평가 데이터 로드
    eval_sets = {}
    for task_name, ds_path in cfg.eval_datasets.items():
        ds = load_from_disk(ds_path)
        split = cfg.eval.get("split", "test")
        n = min(cfg.eval.num_samples, len(ds[split]))
        eval_sets[task_name] = ds[split].select(range(n))
    log.info(f"평가 태스크: {list(eval_sets.keys())}")

    all_results = []

    with mlflow.start_run(run_name="task_arithmetic_multi"):
        mlflow.log_params({
            "base_model": cfg.base_model,
            "vectors": str(vector_names),
            "lambdas": str(list(cfg.lambdas)),
            "eval_tasks": str(list(eval_sets.keys())),
        })

        for lambda_val in cfg.lambdas:
            log.info(f"\n{'='*40}")
            log.info(f"λ = {lambda_val} 적용 중...")

            # base 복원 후 task vector 합산
            model.load_state_dict(base_state)
            apply_multi_vector(model, vectors, lambda_val)

            # 추론을 위해 GPU로 이동
            model.cuda()

            row = {"lambda": lambda_val}
            simpo_cfg = cfg.get("simpo", {})
            beta = simpo_cfg.get("beta", 2.0)
            gamma = simpo_cfg.get("gamma", 0.1)

            for task_name, task_ds in eval_sets.items():
                # Preference-only 태스크: SimPO margin이 주 평가 지표
                # 생성 품질 평가 없이 log prob 기반 margin만 측정 (학습 objective와 동일)
                metrics = compute_simpo_margins(model, tokenizer, task_ds, beta=beta, gamma=gamma)
                row[task_name] = metrics

                lam_str = str(lambda_val).replace(".", "_")
                mlflow.log_metrics({
                    f"{task_name}_simpo_margin_l{lam_str}": metrics["simpo_margin_mean"],
                    f"{task_name}_simpo_loss_l{lam_str}": metrics["simpo_loss"],
                    f"{task_name}_simpo_pos_rate_l{lam_str}": metrics["simpo_margin_positive_rate"],
                })
                log.info(
                    f"  [{task_name}] SimPO margin: {metrics['simpo_margin_mean']:.3f}"
                    f" | pos_rate: {metrics['simpo_margin_positive_rate']:.3f}"
                    f" | SimPO loss: {metrics['simpo_loss']:.4f}"
                )

            all_results.append(row)

            # 다음 λ를 위해 CPU로 복귀
            model.cpu()
            torch.cuda.empty_cache()

        # 결과 요약
        log.info("\n--- λ 스윕 결과 요약 ---")
        task_names = list(eval_sets.keys())
        for r in all_results:
            parts = [
                f"{t} margin={r[t]['simpo_margin_mean']:.3f} pos={r[t]['simpo_margin_positive_rate']:.2f}"
                for t in task_names
            ]
            log.info(f"  λ={r['lambda']:.1f} | " + " | ".join(parts))

        # 태스크별 최적 λ (SimPO margin 기준)
        for task_name in task_names:
            best = max(all_results, key=lambda x: x[task_name]["simpo_margin_mean"])
            log.info(f"최적 λ [{task_name}]: {best['lambda']} (margin: {best[task_name]['simpo_margin_mean']:.3f})")
            mlflow.log_metric(f"best_lambda_{task_name}", best["lambda"])
            mlflow.log_metric(f"best_simpo_margin_{task_name}", best[task_name]["simpo_margin_mean"])


if __name__ == "__main__":
    main()
