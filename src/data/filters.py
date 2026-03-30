"""
데이터 필터링 유틸리티
- HH-RLHF 파싱 (harmlessness)
- UltraFeedback 샘플링 (helpfulness)
- SimPO 포맷 변환
"""

import re
from typing import Optional


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


def parse_hh_conversation(text: str) -> tuple[str, str]:
    """
    HH-RLHF 대화 형식 파싱:
    "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ..."
    → (마지막 Human 발화, 마지막 Assistant 응답)
    """
    # Human/Assistant 턴 분리
    turns = re.split(r'\n\nHuman: |\n\nAssistant: ', text)
    # 첫 빈 토큰 제거
    turns = [t.strip() for t in turns if t.strip()]

    # 대화 구조: [Human, Assistant, Human, Assistant, ...]
    # 마지막 Human → prompt, 마지막 Assistant → response
    human_turns = []
    assistant_turns = []

    # text가 "\n\nHuman:"으로 시작하므로 홀수 인덱스=Human, 짝수 인덱스+1=Assistant
    # 더 안전하게: 원문에서 직접 파싱
    human_pattern = re.findall(r'\n\nHuman: (.*?)(?=\n\nAssistant:|\Z)', text, re.DOTALL)
    assistant_pattern = re.findall(r'\n\nAssistant: (.*?)(?=\n\nHuman:|\Z)', text, re.DOTALL)

    if not human_pattern or not assistant_pattern:
        return "", ""

    last_human = human_pattern[-1].strip()
    last_assistant = assistant_pattern[-1].strip()
    return last_human, last_assistant


def format_for_simpo(example: dict) -> Optional[dict]:
    """
    UltraFeedback binarized → SimPO 학습용 형식으로 변환
    반환값이 None이면 해당 샘플 제외
    """
    prompt = example.get("prompt", "")
    chosen = get_response_text(example.get("chosen", ""))
    rejected = get_response_text(example.get("rejected", ""))

    if not prompt or not chosen or not rejected:
        return {"prompt": None, "chosen": None, "rejected": None}
    if chosen == rejected:
        return {"prompt": None, "chosen": None, "rejected": None}

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def format_hh_for_simpo(example: dict) -> dict:
    """
    HH-RLHF 대화 형식 → SimPO 학습용 형식으로 변환
    chosen/rejected 각각에서 마지막 Human/Assistant 턴 추출
    """
    chosen_text = example.get("chosen", "")
    rejected_text = example.get("rejected", "")

    prompt_c, chosen_resp = parse_hh_conversation(chosen_text)
    prompt_r, rejected_resp = parse_hh_conversation(rejected_text)

    # chosen과 rejected는 같은 대화 맥락을 공유하므로 prompt는 동일해야 함
    prompt = prompt_c or prompt_r

    if not prompt or not chosen_resp or not rejected_resp:
        return {"prompt": None, "chosen": None, "rejected": None}
    if chosen_resp == rejected_resp:
        return {"prompt": None, "chosen": None, "rejected": None}

    return {
        "prompt": prompt,
        "chosen": chosen_resp,
        "rejected": rejected_resp,
    }
