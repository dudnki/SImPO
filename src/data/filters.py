"""
데이터 필터링 유틸리티
- 키워드 기반 태스크 분류
- JSON 품질 검증
- Hard Negative 감지
"""

import json
import re
from typing import Optional


# 태스크별 키워드 맵
TASK_KEYWORDS = {
    "json": [
        "json", "format", "structured output", "key-value",
        "dictionary", "serialize", "schema", "parse json",
    ],
    "math": [
        "calculate", "solve", "equation", "math", "arithmetic",
        "algebra", "geometry",
    ],
    "code": [
        "write code", "python", "function", "implement", "algorithm",
        "debug", "script",
    ],
    "summarize": [
        "summarize", "summary", "tldr", "brief", "shorten",
    ],
}


def is_task(example: dict, task: str) -> bool:
    """주어진 태스크에 해당하는 샘플인지 키워드로 판별"""
    keywords = TASK_KEYWORDS.get(task, [])
    text = " ".join([
        example.get("prompt", ""),
        example.get("instruction", ""),
        example.get("input", ""),
    ]).lower()
    return any(kw in text for kw in keywords)


def extract_json_blocks(text: str) -> list[str]:
    """텍스트에서 JSON 블록 후보를 모두 추출"""
    candidates = []

    # ```json ... ``` 블록
    fenced = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    candidates.extend(fenced)

    # 중괄호/대괄호로 시작하는 standalone JSON
    inline = re.findall(r'(\{[\s\S]*?\}|\[[\s\S]*?\])', text)
    candidates.extend(inline)

    return candidates


def is_valid_json_response(text: str) -> bool:
    """응답 텍스트에 유효한 JSON이 포함되어 있는지 확인"""
    for candidate in extract_json_blocks(text):
        try:
            json.loads(candidate)
            return True
        except (json.JSONDecodeError, ValueError):
            continue
    return False


def is_hard_negative(example: dict) -> bool:
    """
    Hard Negative 조건:
    chosen은 valid JSON 포함, rejected는 JSON 없거나 파싱 실패
    """
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")
    return is_valid_json_response(chosen) and not is_valid_json_response(rejected)


def get_response_text(value) -> str:
    """
    HuggingFace 데이터셋의 chosen/rejected 필드는
    str 또는 list[dict] 형태일 수 있음 — 통일해서 반환
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # [{"role": "assistant", "content": "..."}] 형태
        for turn in reversed(value):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                return turn.get("content", "")
    return ""


def format_for_simpo(example: dict) -> Optional[dict]:
    """
    UltraFeedback binarized → SimPO 학습용 형식으로 변환
    반환값이 None이면 해당 샘플 제외
    """
    prompt = example.get("prompt", "")
    chosen = get_response_text(example.get("chosen", ""))
    rejected = get_response_text(example.get("rejected", ""))

    if not prompt or not chosen or not rejected:
        return None
    if chosen == rejected:
        return None

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
