# configs/ — Hydra 설정 파일

## 이 폴더가 하는 일
각 스크립트의 하이퍼파라미터를 YAML 파일로 관리한다.
코드를 수정하지 않고 설정만 바꿔서 다양한 실험을 돌릴 수 있다.

---

## 핵심 개념

### Hydra란?
Facebook Research가 만든 설정 관리 프레임워크.
YAML 파일의 설정을 Python에서 `cfg.model.name` 형태로 바로 접근할 수 있다.

**CLI에서 설정 오버라이드**:
```bash
# YAML 기본값 대신 다른 값 사용
python scripts/train_simpo.py simpo.beta=3.0 model.lora_r=32

# Orthogonal SimPO 학습 (tau_prev 경로 지정)
python scripts/train_simpo.py \
  data.path=data/processed/math \
  training.output_dir=outputs/adapters/math_simpo_orth \
  orthogonal.tau_prev_path=data/task_vectors/json_simpo_ab.pt \
  orthogonal.alpha=0.1
```
코드를 전혀 안 건드리고 실험 조건을 바꿀 수 있어서 유용하다.

**자동 로그 저장**:
Hydra는 실행할 때마다 자동으로 `outputs/날짜/시간/` 폴더를 만들고 로그를 저장한다.

---

## 설정 파일별 역할

### train_simpo.yaml
SimPO 학습에 필요한 모든 설정.

| 파라미터 | 값 | 의미 |
|---|---|---|
| `model.lora_r` | 16 | LoRA rank. 클수록 표현력↑, 메모리↑ |
| `model.lora_alpha` | 16 | LoRA 스케일링. alpha/r = 1.0 |
| `simpo.beta` | 2.0 | 보상 마진 강조 강도 |
| `simpo.gamma` | 0.5 | 목표 보상 마진 |
| `training.gradient_accumulation_steps` | 8 | 유효 배치 크기 = 1×8 = 8 |
| `training.eval_strategy` | "no" | 중간 평가 비활성화 (OOM 방지) |
| `orthogonal.tau_prev_path` | null | 이전 태스크 AB 행렬 경로. null이면 일반 SimPO |
| `orthogonal.alpha` | 0.1 | Orthogonal Loss 가중치 |

### apply_arithmetic.yaml
fp16 베이스 모델에 task vector를 직접 합산하는 평가 설정.

| 파라미터 | 값 | 의미 |
|---|---|---|
| `base_model` | fp16 모델 경로 | 4-bit 아닌 fp16 필수 (weight 합산 때문) |
| `vector_paths` | .pt 파일 목록 | 합산할 task vector들 |
| `eval_datasets` | 태스크명: 데이터 경로 | 태스크별 평가 데이터 |
| `lambdas` | [0.0~1.5] | 테스트할 λ 값 목록 |
| `eval.system_prompt` | "Always respond in valid JSON..." | 평가 시 시스템 프롬프트 |

### extract_task_vector.yaml
태스크 벡터 추출 설정.

| 파라미터 | 값 | 의미 |
|---|---|---|
| `save_lora_ab` | false | true로 설정 시 A,B raw 행렬도 함께 저장 |
| `lora_ab_output` | .pt 경로 | tau_prev용 AB 행렬 저장 경로 |

---

## MLflow란?
실험 결과(메트릭, 파라미터, 모델)를 추적하는 오픈소스 플랫폼.
우리는 SQLite 파일(`mlflow.db`)을 백엔드로 사용한다.

```bash
# UI 실행 (가상환경 활성화 후)
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
# → http://localhost:5000 에서 시각적으로 확인
```

각 스크립트 실행마다 자동으로 메트릭이 기록된다:
- 학습: loss, accuracy, reward_margin, orthogonal_loss (Orthogonal SimPO 시)
- task vector 추출: num_modified_layers, vector_size_mb
- λ 스윕: 태스크별 json_validity_rate, json_key_match_rate
