"""
prepare_dataset.py
==================
Preference-only 태스크별 선호도 데이터를 준비합니다.

태스크:
  helpfulness  — UltraFeedback (일반 유용성 선호도)
  harmlessness — HH-RLHF harmless split (안전성 선호도)

사용법:
    python scripts/prepare_dataset.py --task helpfulness
    python scripts/prepare_dataset.py --task harmlessness
    python scripts/prepare_dataset.py --task helpfulness --n_samples 3000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset, Dataset
from src.data.filters import format_for_simpo, format_hh_for_simpo

import mlflow

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
SAVE_DIR = Path("data/processed")


def load_helpfulness(n_samples: int) -> Dataset:
    """
    UltraFeedback binarized — 일반 유용성 선호도
    키워드 필터 없이 전체 사용: 데이터 자체가 이미 품질 기반 선호도 쌍
    """
    print("Loading UltraFeedback binarized (helpfulness)...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    if n_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(n_samples))
    print(f"  로드 완료: {len(ds):,}개")
    return ds


def load_hh_helpful(n_samples: int) -> Dataset:
    """
    Anthropic HH-RLHF helpful split — 멀티턴 대화 유용성 선호도
    형식: 멀티턴 대화 문자열에서 마지막 Human/Assistant 턴 추출
    """
    print("Loading HH-RLHF helpful split...")
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    if n_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(n_samples))
    print(f"  로드 완료: {len(ds):,}개")
    return ds


def load_coding(n_samples: int) -> Dataset:
    """
    UltraFeedback binarized — 코딩 태스크 선호도
    prompt에 코딩 관련 키워드가 있는 샘플만 필터링
    """
    CODING_KEYWORDS = [
        'write a python', 'write python', 'python script', 'python code',
        'write a javascript', 'javascript code', 'write a function',
        'write a program', 'write code', 'implement a', 'implement the',
        'debug this', 'fix this code', 'write a class', 'write a script',
        'write a sql', 'write an sql', 'write a bash', 'write a shell',
        'in python', 'in javascript', 'in java', 'in c++',
    ]

    print("Loading UltraFeedback coding subset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    ds = ds.filter(
        lambda x: any(kw in x["prompt"].lower() for kw in CODING_KEYWORDS),
        desc="coding 필터링",
    )
    ds = ds.shuffle(seed=42)
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))
    print(f"  로드 완료: {len(ds):,}개")
    return ds


def load_safety(n_samples: int) -> Dataset:
    """
    PKU-SafeRLHF — 안전성 선호도 (GPT-4 레이블)
    safer_response_id 기준으로 chosen/rejected 구성
    안전 응답이 명확히 구분되는 샘플 우선 사용
    """
    print("Loading PKU-SafeRLHF (safety)...")
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    ds = ds.shuffle(seed=42)

    examples = []
    for item in ds:
        safe_0 = item["is_response_0_safe"]
        safe_1 = item["is_response_1_safe"]

        # 한쪽은 안전, 한쪽은 불안전인 샘플만 사용 — 신호가 명확함
        if safe_0 == safe_1:
            continue

        # 안전한 응답 = chosen, 불안전한 응답 = rejected
        if safe_0 and not safe_1:
            chosen = item["response_0"]
            rejected = item["response_1"]
        else:
            chosen = item["response_1"]
            rejected = item["response_0"]

        prompt = item["prompt"]
        if not prompt or not chosen or not rejected:
            continue
        if chosen == rejected:
            continue

        examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

        if len(examples) >= n_samples:
            break

    print(f"  로드 완료: {len(examples):,}개")
    return Dataset.from_list(examples)


TASK_CONFIG = {
    "helpfulness": {
        "loader": load_helpfulness,
        "formatter": format_for_simpo,
        "description": "UltraFeedback 일반 유용성 선호도",
        "default_n": 5000,
    },
    "hh_helpful": {
        "loader": load_hh_helpful,
        "formatter": format_hh_for_simpo,
        "description": "HH-RLHF 멀티턴 대화 유용성 선호도",
        "default_n": 5000,
    },
    "safety": {
        "loader": load_safety,
        "formatter": format_for_simpo,
        "description": "PKU-SafeRLHF 안전성 선호도 (GPT-4 레이블)",
        "default_n": 5000,
    },
    "coding": {
        "loader": load_coding,
        "formatter": format_for_simpo,
        "description": "UltraFeedback 코딩 선호도 (GPT-4 레이블)",
        "default_n": 5000,
    },
}


def run(task: str, n_samples: int):
    if task not in TASK_CONFIG:
        print(f"[오류] 지원하지 않는 태스크: {task}")
        print(f"       지원 태스크: {list(TASK_CONFIG.keys())}")
        return

    cfg = TASK_CONFIG[task]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("simpo_task_arithmetic")

    with mlflow.start_run(run_name=f"prepare_{task}"):
        mlflow.log_params({
            "task": task,
            "n_samples_requested": n_samples,
            "source": cfg["description"],
        })

        print(f"\n{'='*50}")
        print(f"태스크: {task} — {cfg['description']}")
        print(f"{'='*50}")

        # 1. 데이터 로드
        raw = cfg["loader"](n_samples)
        mlflow.log_metric("raw_samples", len(raw))

        # 2. SimPO 형식 변환
        formatter = cfg["formatter"]
        formatted = raw.map(
            formatter,
            remove_columns=raw.column_names,
            load_from_cache_file=False,
            desc="SimPO 형식 변환",
        )

        # 3. None 제거
        formatted = formatted.filter(
            lambda x: x["prompt"] is not None and x["chosen"] is not None and x["rejected"] is not None,
            load_from_cache_file=False,
            desc="유효 샘플 필터링",
        )
        mlflow.log_metric("valid_samples", len(formatted))
        print(f"유효 샘플: {len(formatted):,}개 (변환 전 {len(raw):,}개)")

        if len(formatted) == 0:
            print("[오류] 유효한 샘플이 없습니다.")
            return

        # 4. train/test 분할 (9:1)
        split = formatted.train_test_split(test_size=0.1, seed=42)
        mlflow.log_metric("train_samples", len(split["train"]))
        mlflow.log_metric("test_samples", len(split["test"]))
        print(f"  train: {len(split['train']):,}  /  test: {len(split['test']):,}")

        # 5. 저장
        save_path = SAVE_DIR / task
        save_path.mkdir(parents=True, exist_ok=True)
        split.save_to_disk(str(save_path))
        mlflow.log_param("save_path", str(save_path))
        print(f"저장 완료: {save_path}")

        # 샘플 미리보기
        print("\n--- 샘플 미리보기 ---")
        for i, sample in enumerate(split["train"].select(range(min(2, len(split["train"]))))):
            print(f"\n[{i+1}] prompt:   {sample['prompt'][:120]}...")
            print(f"     chosen:   {sample['chosen'][:100]}...")
            print(f"     rejected: {sample['rejected'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preference-only 태스크별 선호도 데이터 준비")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_CONFIG.keys()),  # helpfulness, hh_helpful
        help="준비할 태스크",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="로드할 최대 샘플 수 (기본값: 태스크별 기본값 사용)",
    )
    args = parser.parse_args()

    n = args.n_samples if args.n_samples is not None else TASK_CONFIG[args.task]["default_n"]
    run(args.task, n)
