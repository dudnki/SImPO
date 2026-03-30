"""태스크별 평가 지표 (JSON 포맷팅, 수학 정확도, SimPO reward margin)"""
import json
import re
from typing import Optional

import torch
import torch.nn.functional as F


def extract_json(text: str) -> Optional[str]:
    """응답에서 JSON 블록 추출"""
    # ```json ... ``` 블록 우선
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    # ``` ... ``` 블록
    pattern = r"```\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    # 중괄호/대괄호로 시작하는 raw JSON
    pattern = r"(\{[\s\S]*\}|\[[\s\S]*\])"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    return None


def is_valid_json(text: str) -> bool:
    """유효한 JSON 포함 여부"""
    candidate = extract_json(text)
    if candidate is None:
        return False
    try:
        json.loads(candidate)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def json_validity_rate(responses: list[str]) -> float:
    """JSON 유효율 (0.0 ~ 1.0)"""
    if not responses:
        return 0.0
    valid = sum(1 for r in responses if is_valid_json(r))
    return valid / len(responses)


def json_key_match_rate(responses: list[str], references: list[str]) -> float:
    """chosen 응답과 키 집합 일치율"""
    if not responses:
        return 0.0

    matches = 0
    for resp, ref in zip(responses, references):
        resp_json = extract_json(resp)
        ref_json = extract_json(ref)
        if resp_json is None or ref_json is None:
            continue
        try:
            resp_obj = json.loads(resp_json)
            ref_obj = json.loads(ref_json)
            if isinstance(resp_obj, dict) and isinstance(ref_obj, dict):
                resp_keys = set(resp_obj.keys())
                ref_keys = set(ref_obj.keys())
                if ref_keys:
                    matches += len(resp_keys & ref_keys) / len(ref_keys)
        except (json.JSONDecodeError, ValueError):
            continue

    return matches / len(responses)


def extract_final_number(text: str) -> Optional[str]:
    """GSM8K 스타일 정답 추출: '#### 72' → '72', 없으면 마지막 숫자"""
    m = re.search(r'####\s*([\d,.\-]+)', text)
    if m:
        return m.group(1).strip().replace(',', '')
    # #### 없으면 응답의 마지막 숫자
    nums = re.findall(r'[\d,]+(?:\.\d+)?', text)
    return nums[-1].replace(',', '') if nums else None


def math_answer_accuracy(responses: list[str], references: list[str]) -> float:
    """GSM8K 정답 일치율: 응답에서 추출한 숫자 vs 정답 숫자"""
    if not responses:
        return 0.0
    correct = 0
    for resp, ref in zip(responses, references):
        pred = extract_final_number(resp)
        gold = extract_final_number(ref)
        if pred is not None and gold is not None and pred == gold:
            correct += 1
    return correct / len(responses)


def compute_avg_logp(model, tokenizer, prompt: str, response: str, device: str = "cuda") -> float:
    """
    응답 토큰들의 평균 log probability 계산.
    SimPO와 동일한 방식: log P(y|x) / |y|
    """
    messages_full = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    messages_prompt = [
        {"role": "user", "content": prompt},
    ]

    def _to_tensor(encoded, dev):
        if isinstance(encoded, torch.Tensor):
            return encoded.to(dev)
        return encoded["input_ids"].to(dev)

    full_ids = _to_tensor(
        tokenizer.apply_chat_template(messages_full, add_generation_prompt=False, return_tensors="pt"),
        device,
    )
    prompt_ids = _to_tensor(
        tokenizer.apply_chat_template(messages_prompt, add_generation_prompt=True, return_tensors="pt"),
        device,
    )
    prompt_len = prompt_ids.shape[-1]

    with torch.no_grad():
        logits = model(full_ids).logits  # (1, seq_len, vocab_size)

    # logits[i] → token[i+1] 예측
    shift_logits = logits[0, :-1, :]       # (seq_len-1, vocab)
    shift_labels = full_ids[0, 1:]         # (seq_len-1,)

    vocab_size = shift_logits.shape[-1]
    shift_labels = shift_labels.clamp(0, vocab_size - 1)  # OOB 토큰 클램핑

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs[range(len(shift_labels)), shift_labels]

    # 응답 토큰 구간만 추출 (prompt 이후), 범위 초과 방지
    start = min(prompt_len - 1, len(token_log_probs) - 1)
    response_log_probs = token_log_probs[start:]
    if len(response_log_probs) == 0:
        return 0.0
    return response_log_probs.mean().item()


def compute_simpo_margins(
    model,
    tokenizer,
    dataset,
    beta: float = 2.0,
    gamma: float = 0.1,
    device: str = "cuda",
) -> dict:
    """
    SimPO reward margin 계산: β*(avg_logP_chosen - avg_logP_rejected) - γ
    학습 때와 동일한 objective로 평가.
    """
    margins = []
    for ex in dataset:
        logp_c = compute_avg_logp(model, tokenizer, ex["prompt"], ex["chosen"], device)
        logp_r = compute_avg_logp(model, tokenizer, ex["prompt"], ex["rejected"], device)
        margin = beta * (logp_c - logp_r) - gamma
        margins.append(margin)

    avg_margin = sum(margins) / len(margins) if margins else 0.0
    pos_rate = sum(1 for m in margins if m > 0) / len(margins) if margins else 0.0

    # SimPO loss = -log σ(margin)  (학습 때와 동일한 objective)
    margin_tensor = torch.tensor(margins)
    avg_loss = F.softplus(-margin_tensor).mean().item()  # numerically stable -log sigmoid

    return {
        "simpo_margin_mean": avg_margin,
        "simpo_margin_positive_rate": pos_rate,
        "simpo_loss": avg_loss,
    }


def compute_metrics(responses: list[str], references: list[str], task: str = "json") -> dict:
    """태스크별 지표 계산 (task: 'json' | 'gsm8k')"""
    if task == "gsm8k":
        acc = math_answer_accuracy(responses, references)
        return {
            "math_accuracy": acc,
            "n_samples": len(responses),
            "n_correct": int(acc * len(responses)),
        }
    # default: json
    return {
        "json_validity_rate": json_validity_rate(responses),
        "json_key_match_rate": json_key_match_rate(responses, references),
        "n_samples": len(responses),
        "n_valid": sum(1 for r in responses if is_valid_json(r)),
    }
