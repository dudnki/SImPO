"""
prepare_dataset.py
==================
공개 데이터셋(UltraFeedback 등)에서 태스크별 선호도 데이터를 추출·저장합니다.

사용법:
    python scripts/prepare_dataset.py --task json
    python scripts/prepare_dataset.py --task json --hard_negative_only
    python scripts/prepare_dataset.py --task json --also_use openhermes
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset, concatenate_datasets, Dataset
from src.data.filters import is_task, is_hard_negative, format_for_simpo

import mlflow

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
SAVE_DIR = Path("data/processed")


def load_ultrafeedback(split: str = "train_prefs") -> Dataset:
    print("Loading UltraFeedback binarized...")
    return load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split=split,
    )


def load_openhermes() -> Dataset:
    """OpenHermes 2.5 — chosen/rejected 구조로 변환 필요"""
    print("Loading OpenHermes 2.5...")
    raw = load_dataset("teknium/OpenHermes-2.5", split="train")

    # OpenHermes는 단일 응답 데이터셋이므로 pseudo-rejected 생성:
    # 원본 응답을 chosen, 앞부분을 잘라낸 버전을 rejected으로 사용
    def make_pair(example):
        instruction = example.get("instruction", "")
        output = example.get("output", "")
        if not instruction or not output or len(output) < 50:
            return {"prompt": None, "chosen": None, "rejected": None}
        # 뒷부분 30%를 제거한 것을 rejected으로
        cutoff = int(len(output) * 0.7)
        return {
            "prompt": instruction,
            "chosen": output,
            "rejected": output[:cutoff] + "\n[응답이 불완전합니다]",
        }

    processed = raw.map(make_pair, remove_columns=raw.column_names)
    return processed.filter(lambda x: x["prompt"] is not None)


LOADERS = {
    "ultrafeedback": load_ultrafeedback,
    "openhermes": load_openhermes,
}


def run(task: str, hard_negative_only: bool, also_use: list[str]):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_name=f"prepare_{task}"):
        mlflow.log_params({
            "task": task,
            "hard_negative_only": hard_negative_only,
            "sources": ["ultrafeedback"] + also_use,
        })

        # 1. 데이터 로드
        datasets = [load_ultrafeedback()]
        for source in also_use:
            if source in LOADERS:
                datasets.append(LOADERS[source]())
            else:
                print(f"[경고] 알 수 없는 소스: {source}, 건너뜁니다.")

        combined = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
        mlflow.log_metric("raw_total", len(combined))
        print(f"전체 샘플: {len(combined):,}")

        # 2. 태스크 필터
        task_filtered = combined.filter(
            lambda x: is_task(x, task),
            num_proc=4,
            desc=f"'{task}' 태스크 필터링",
        )
        mlflow.log_metric("after_task_filter", len(task_filtered))
        print(f"'{task}' 관련 샘플: {len(task_filtered):,}")

        # 3. Hard Negative 필터 (선택)
        if hard_negative_only:
            task_filtered = task_filtered.filter(
                is_hard_negative,
                desc="Hard Negative 필터링",
            )
            mlflow.log_metric("after_hard_negative_filter", len(task_filtered))
            print(f"Hard Negative 샘플: {len(task_filtered):,}")

        # 4. SimPO 형식으로 변환
        formatted = task_filtered.map(
            format_for_simpo,
            remove_columns=task_filtered.column_names,
            desc="SimPO 형식 변환",
        )
        formatted = formatted.filter(lambda x: x["prompt"] is not None)
        mlflow.log_metric("final_samples", len(formatted))
        print(f"최종 저장 샘플: {len(formatted):,}")

        if len(formatted) == 0:
            print("[오류] 필터링 후 샘플이 없습니다. 키워드나 필터 조건을 확인하세요.")
            return

        # 5. train/validation 분할 (9:1)
        split = formatted.train_test_split(test_size=0.1, seed=42)
        print(f"  train: {len(split['train']):,}  /  validation: {len(split['test']):,}")

        # 6. 저장
        suffix = "_hard" if hard_negative_only else ""
        save_path = SAVE_DIR / f"{task}{suffix}"
        split.save_to_disk(str(save_path))
        mlflow.log_param("save_path", str(save_path))
        print(f"저장 완료: {save_path}")

        # 샘플 미리보기
        print("\n--- 샘플 미리보기 ---")
        for i, sample in enumerate(split["train"].select(range(min(2, len(split["train"]))))):
            print(f"\n[{i+1}] prompt: {sample['prompt'][:100]}...")
            print(f"     chosen:   {sample['chosen'][:80]}...")
            print(f"     rejected: {sample['rejected'][:80]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="태스크별 선호도 데이터 준비")
    parser.add_argument(
        "--task",
        type=str,
        default="json",
        choices=["json", "math", "code", "summarize"],
        help="추출할 태스크 종류",
    )
    parser.add_argument(
        "--hard_negative_only",
        action="store_true",
        help="chosen=valid JSON, rejected=invalid JSON인 샘플만 사용",
    )
    parser.add_argument(
        "--also_use",
        nargs="*",
        default=[],
        choices=list(LOADERS.keys()),
        help="UltraFeedback 외 추가 데이터셋",
    )
    args = parser.parse_args()
    run(args.task, args.hard_negative_only, args.also_use)
