"""
HumanEval pass@1 평가 스크립트
Task Arithmetic 합성 모델의 코딩 능력 보존을 측정

사용법:
    # coding 단독 (기준선)
    python scripts/eval_humaneval.py \
        vector_paths=[data/task_vectors/coding_simpo.pt] \
        run_name=coding_only

    # helpfulness + coding baseline 합성
    python scripts/eval_humaneval.py \
        "vector_paths=[data/task_vectors/helpfulness_simpo.pt,data/task_vectors/coding_simpo.pt]" \
        run_name=baseline_merged

    # helpfulness + coding orth 합성
    python scripts/eval_humaneval.py \
        "vector_paths=[data/task_vectors/helpfulness_simpo.pt,data/task_vectors/coding_simpo_orth.pt]" \
        run_name=orth_merged
"""
import logging
import sys
import json
import tempfile
import subprocess
from pathlib import Path

import torch
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from human_eval.data import read_problems, write_jsonl

log = logging.getLogger(__name__)

HUMANEVAL_PROBLEMS = read_problems()


def load_fp16_model(base_model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def apply_vectors(model, vector_paths: list, lambda_val: float):
    with torch.no_grad():
        for vp in vector_paths:
            vec = torch.load(vp, map_location="cpu", weights_only=True)
            for name, param in model.named_parameters():
                if name in vec:
                    param.data.add_(lambda_val * vec[name].to(param.dtype))
            del vec


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def stop_at_function_end(completion: str) -> str:
    """함수 바디가 끝나는 지점에서 자르기"""
    lines = completion.split("\n")
    result = []
    for line in lines:
        if line and not line[0].isspace() and result:
            break
        result.append(line)
    return "\n".join(result)


def evaluate_pass_at_1(model, tokenizer, n_problems: int = 164) -> float:
    problems = list(HUMANEVAL_PROBLEMS.values())[:n_problems]
    samples = []

    model.cuda()
    model.eval()

    for i, problem in enumerate(problems):
        prompt = problem["prompt"]
        completion = generate_completion(model, tokenizer, prompt)
        completion = stop_at_function_end(completion)
        samples.append({
            "task_id": problem["task_id"],
            "completion": completion,
        })
        if (i + 1) % 20 == 0:
            log.info(f"  진행: {i+1}/{n_problems}")

    model.cpu()
    torch.cuda.empty_cache()

    # 임시 파일에 저장 후 human_eval 실행
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        write_jsonl(f.name, samples)
        sample_file = f.name

    result = subprocess.run(
        [sys.executable, "-c", f"""
from human_eval.evaluation import evaluate_functional_correctness
result = evaluate_functional_correctness("{sample_file}")
print(result['pass@1'])
"""],
        capture_output=True, text=True
    )

    if result.returncode != 0 or not result.stdout.strip():
        log.error(f"human_eval subprocess 실패:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        raise RuntimeError(f"evaluate_functional_correctness 실패: {result.stderr[:500]}")

    pass_at_1 = float(result.stdout.strip().split("\n")[-1])
    Path(sample_file).unlink(missing_ok=True)
    return pass_at_1


@hydra.main(config_path="../configs", config_name="apply_arithmetic", version_base=None)
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = cfg.get("run_name", "humaneval_eval")
    lambda_val = float(cfg.get("lambda_val", 1.0))
    n_problems = int(cfg.get("n_problems", 164))
    vector_paths = list(cfg.vector_paths)
    vector_names = [Path(vp).stem for vp in vector_paths]

    log.info(f"벡터: {vector_names}")
    log.info(f"λ={lambda_val}, 문제 수={n_problems}")

    model, tokenizer = load_fp16_model("unsloth/Llama-3.2-3B-Instruct")

    # base 저장
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    # λ=0 (베이스 모델)
    log.info("λ=0.0 평가 중 (베이스)...")
    pass_at_1_base = evaluate_pass_at_1(model, tokenizer, n_problems)
    log.info(f"베이스 pass@1: {pass_at_1_base:.4f}")

    # task vector 적용
    model.load_state_dict(base_state)
    apply_vectors(model, vector_paths, lambda_val)
    log.info(f"λ={lambda_val} 평가 중...")
    pass_at_1_merged = evaluate_pass_at_1(model, tokenizer, n_problems)
    log.info(f"합성 pass@1: {pass_at_1_merged:.4f}")
    log.info(f"변화량: {pass_at_1_merged - pass_at_1_base:+.4f}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "vectors": str(vector_names),
            "lambda_val": lambda_val,
            "n_problems": n_problems,
        })
        mlflow.log_metrics({
            "pass_at_1_base": pass_at_1_base,
            "pass_at_1_merged": pass_at_1_merged,
            "pass_at_1_delta": pass_at_1_merged - pass_at_1_base,
        })


if __name__ == "__main__":
    main()
