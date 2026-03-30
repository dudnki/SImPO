# data/ — 데이터 저장소

## 이 폴더가 하는 일
파이프라인의 각 단계에서 생성되는 데이터 파일들을 저장한다.

---

## 폴더 구조

```
data/
├── raw/              # HuggingFace 원본 다운로드 캐시 (자동 생성)
├── processed/        # 필터링된 선호도 데이터셋
│   └── json/         # JSON 태스크용 데이터
│       ├── train/    # 545개 학습 샘플 (Arrow 형식)
│       └── test/     # 61개 평가 샘플 (Arrow 형식)
└── task_vectors/     # 추출된 태스크 벡터
    └── json_simpo.pt # JSON SimPO 태스크 벡터
```

---

## 핵심 개념

### Arrow 형식이란?
HuggingFace Datasets 라이브러리가 사용하는 컬럼형 데이터 저장 형식.
- 빠른 읽기 속도
- 메모리 매핑 지원 (RAM에 전부 올리지 않고도 접근 가능)
- `load_from_disk()` / `save_to_disk()`로 읽고 씀

### .pt 파일이란?
PyTorch 텐서를 저장하는 형식. `torch.save()` / `torch.load()`로 읽고 씀.
`json_simpo.pt` 파일 내부:
```python
{
    "model.layers.0.self_attn.q_proj.weight": tensor([...]),  # ΔW
    "model.layers.0.self_attn.k_proj.weight": tensor([...]),
    ...
    # 총 196개 레이어
}
```

---

## 주의사항

### .gitignore에서 제외된 이유
- `data/processed/`: Arrow 파일은 수백 MB ~ 수 GB → git에 올리면 안 됨
- `data/task_vectors/`: 태스크 벡터 `.pt` 파일이 5.6 GB → 절대 git에 올리면 안 됨

데이터를 다시 만들려면 스크립트를 재실행:
```bash
python scripts/prepare_dataset.py --task json       # processed/ 재생성
python scripts/extract_task_vector.py               # task_vectors/ 재생성
```

### HuggingFace 캐시 문제
`data/raw/`가 아니라 `~/.cache/huggingface/datasets/`에 캐시가 저장된다.
필터 조건을 바꿨는데 결과가 안 바뀌면 캐시를 직접 삭제해야 할 수 있다:
```bash
rm -rf ~/.cache/huggingface/datasets/
```
