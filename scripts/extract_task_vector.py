"""
extract_task_vector.py
======================
학습된 LoRA 어댑터에서 Task Vector(Δθ)를 추출합니다.

사용법:
    python scripts/extract_task_vector.py
    python scripts/extract_task_vector.py adapter_path=outputs/adapters/json_simpo
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import mlflow
from omegaconf import DictConfig

from src.arithmetic.task_vector import extract_task_vector, extract_lora_ab


@hydra.main(config_path="../configs", config_name="extract_task_vector", version_base=None)
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="extract_task_vector"):
        mlflow.log_params({
            "base_model": cfg.base_model,
            "adapter_path": cfg.adapter_path,
            "vector_output": cfg.vector_output,
        })

        task_vector = extract_task_vector(
            base_model_path=cfg.base_model,
            adapter_path=cfg.adapter_path,
            output_path=cfg.vector_output,
        )

        mlflow.log_metrics({
            "num_modified_layers": len(task_vector),
            "vector_size_mb": sum(
                v.numel() * v.element_size() for v in task_vector.values()
            ) / 1e6,
        })

        print(f"\n추출된 레이어 수: {len(task_vector)}")
        print(f"벡터 크기: {sum(v.numel() * v.element_size() for v in task_vector.values()) / 1e6:.1f} MB")

        if cfg.get("save_lora_ab", False):
            lora_ab = extract_lora_ab(
                adapter_path=cfg.adapter_path,
                output_path=cfg.lora_ab_output,
            )
            mlflow.log_param("lora_ab_output", cfg.lora_ab_output)
            mlflow.log_metric("lora_ab_layers", len(lora_ab))


if __name__ == "__main__":
    main()
