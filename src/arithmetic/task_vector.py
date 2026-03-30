"""
Task Vector 추출 및 산술 연산
θ_new = θ_base + λ * Δθ   (Δθ = B @ A * scaling)

extract_task_vector : ΔW = B@A*scaling 형태로 저장 (apply_arithmetic용)
extract_lora_ab     : A, B 행렬 raw 형태로 저장 (OrthogonalSimPO tau_prev용)
"""

import json
import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_task_vector(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """
    LoRA 가중치에서 직접 Task Vector를 계산.
    ΔW = B @ A * (lora_alpha / r)
    모델을 로드하지 않아도 되므로 빠르고 메모리 효율적.
    """
    # adapter config에서 scaling 계산
    with open(Path(adapter_path) / "adapter_config.json") as f:
        config = json.load(f)
    r = config["r"]
    lora_alpha = config["lora_alpha"]
    scaling = lora_alpha / r
    print(f"LoRA 설정: r={r}, alpha={lora_alpha}, scaling={scaling:.3f}")

    # LoRA 가중치 로드
    print("LoRA 가중치 로드 중...")
    weights = load_file(Path(adapter_path) / "adapter_model.safetensors")

    # A 행렬 키 목록
    lora_A_keys = [k for k in weights if k.endswith("lora_A.weight")]

    task_vector = {}
    for a_key in lora_A_keys:
        b_key = a_key.replace("lora_A.weight", "lora_B.weight")
        if b_key not in weights:
            continue

        A = weights[a_key].float()  # (r, in_features)
        B = weights[b_key].float()  # (out_features, r)
        delta = (B @ A * scaling).to(dtype)

        # base model 파라미터 이름으로 변환
        # "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        # → "model.layers.0.self_attn.q_proj.weight"
        param_name = (
            a_key
            .removeprefix("base_model.model.")
            .replace(".lora_A.weight", ".weight")
        )
        task_vector[param_name] = delta

    total_params = sum(v.numel() for v in task_vector.values())
    size_mb = sum(v.numel() * v.element_size() for v in task_vector.values()) / 1e6
    print(f"추출 완료: {len(task_vector)}개 레이어, {total_params:,} 파라미터, {size_mb:.1f} MB")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(task_vector, output_path)
    print(f"저장: {output_path}")

    return task_vector


def extract_lora_ab(
    adapter_path: str,
    output_path: str,
) -> dict:
    """
    LoRA A, B 행렬을 raw 형태로 저장 (OrthogonalSimPO의 tau_prev용).
    형식: {module_name: {"A": tensor(r, in), "B": tensor(out, r)}}

    module_name은 model.named_modules()의 키와 일치:
    "base_model.model.model.layers.X.self_attn.q_proj"
    (extract_task_vector와 달리 base_model 접두사를 유지)
    """
    weights = load_file(Path(adapter_path) / "adapter_model.safetensors")
    lora_A_keys = [k for k in weights if k.endswith("lora_A.weight")]

    lora_ab = {}
    for a_key in lora_A_keys:
        b_key = a_key.replace("lora_A.weight", "lora_B.weight")
        if b_key not in weights:
            continue

        A = weights[a_key].float()  # (r, in_features)
        B = weights[b_key].float()  # (out_features, r)

        # "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        # → "base_model.model.model.layers.0.self_attn.q_proj"
        module_name = a_key.removesuffix(".lora_A.weight")

        lora_ab[module_name] = {"A": A, "B": B}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_ab, output_path)
    print(f"tau_prev 저장: {output_path} ({len(lora_ab)} 레이어)")
    return lora_ab


def apply_task_vector(
    base_model_path: str,
    vector_path: str,
    lambda_val: float,
    output_path: str,
    dtype: torch.dtype = torch.float16,
):
    """
    θ_new = θ_base + λ * Δθ 를 적용하고 full model 저장.
    """
    print(f"Base 모델 로드 중 (λ={lambda_val})...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print("Task Vector 로드 중...")
    task_vector = torch.load(vector_path, map_location="cpu", weights_only=True)

    print(f"Task Arithmetic 적용 중 (λ={lambda_val})...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in task_vector:
                param.data += lambda_val * task_vector[name].to(param.dtype)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"저장: {output_path}")

    del model, task_vector
    torch.cuda.empty_cache()


def lambda_sweep(
    base_model_path: str,
    vector_path: str,
    lambdas: list[float],
    output_base_dir: str,
):
    """여러 λ 값으로 병합 모델을 일괄 생성"""
    for lam in lambdas:
        lam_str = str(lam).replace(".", "_")
        output_path = str(Path(output_base_dir) / f"lam{lam_str}")
        print(f"\n--- λ={lam} ---")
        apply_task_vector(base_model_path, vector_path, lam, output_path)
