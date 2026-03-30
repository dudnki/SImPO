"""
LoRA 어댑터 직접 추론 vs Task Arithmetic 비교
4bit 모델 + LoRA 어댑터로 직접 추론하여 JSON validity 측정
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import compute_metrics


ADAPTER_CONFIGS = [
    {
        "name": "json_hard LoRA (4bit)",
        "adapter_path": "outputs/adapters/json_hard_simpo",
        "task": "json",
        "dataset": "data/processed/json",
        "system_prompt": "You are a helpful assistant. Always respond in valid JSON format.",
    },
    {
        "name": "gsm8k LoRA (4bit)",
        "adapter_path": "outputs/adapters/gsm8k_simpo",
        "task": "gsm8k",
        "dataset": "data/processed/gsm8k",
        "system_prompt": "You are a helpful math assistant. Solve the problem step by step and end with '#### <answer>'.",
    },
]

NUM_SAMPLES = 61
MAX_NEW_TOKENS = 256
BETA = 2.0
GAMMA = 0.1


def generate_responses(model, tokenizer, prompts, system_prompt):
    model.eval()
    responses = []
    for prompt in tqdm(prompts, desc="추론 중", leave=False):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.cuda()
        else:
            input_ids = encoded["input_ids"].cuda()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


def main():
    print("=" * 60)
    print("LoRA 어댑터 직접 추론 평가")
    print("(Task Arithmetic과 비교용)")
    print("=" * 60)

    for cfg in ADAPTER_CONFIGS:
        print(f"\n--- {cfg['name']} ---")
        print(f"어댑터: {cfg['adapter_path']}")

        # fp16 base + LoRA 어댑터 로드 후 merge (Unsloth 패치 없이)
        print("  fp16 base 모델 로드 중...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg["adapter_path"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("  LoRA 어댑터 병합 중...")
        model = PeftModel.from_pretrained(base_model, cfg["adapter_path"])
        model = model.merge_and_unload()  # LoRA → fp16 weight에 병합
        model.eval()

        ds = load_from_disk(cfg["dataset"])
        test_ds = ds["test"].select(range(min(NUM_SAMPLES, len(ds["test"]))))

        prompts = [ex["prompt"] for ex in test_ds]
        refs = [ex["chosen"] for ex in test_ds]

        responses = generate_responses(model, tokenizer, prompts, cfg["system_prompt"])
        metrics = compute_metrics(responses, refs, task=cfg["task"])

        if cfg["task"] == "gsm8k":
            print(f"  수학 정확도:    {metrics['math_accuracy']:.3f} ({metrics['n_correct']}/{metrics['n_samples']})")
        else:
            print(f"  JSON 유효율:   {metrics['json_validity_rate']:.3f}")
            print(f"  키 일치율:     {metrics['json_key_match_rate']:.3f}")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Task Arithmetic 결과 (참고, λ=1.0)")
    print("  json  → JSON 유효율: 0.459 | SimPO loss: 0.6302")
    print("  gsm8k → 수학 정확도: 0.836 | SimPO loss: 0.0844")
    print("=" * 60)


if __name__ == "__main__":
    main()
