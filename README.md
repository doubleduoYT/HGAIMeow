# HGAI v10 Practical Standalone Engine

HGAI v10은 외부 LLM을 런타임 의존성으로 사용하지 않는 자체 한국어 고양이 AI 엔진이다.

## 핵심 변경점

- `phone` 프리셋 완전 제거. 기본은 `main`, 더 큰 실험용은 `large`.
- 최종 실행 시 Qwen/SmolLM/ChatGPT 같은 외부 모델이 필요하지 않음.
- 외부 모델은 `teacher_reference.py`로 **검토용 제안 데이터**를 만들 때만 선택적으로 참고 가능하며 자동 병합되지 않음.
- 기존 12,568개 데이터 + v10 추가 데이터 + 의미 보존형 질문 변형을 학습 시 확장.
- 현재 effective dataset은 약 10만 쌍이며 `build_dataset.py`로 재생성 가능.
- `Casualties: Unknown` 요청 문답과 한글/영문 표현 변형 추가.
- 동일 질문의 첫 답만 고정 반환하던 동작 제거.
- 한국어/영문 동의어 정규화 + 문자 n-gram/어휘 기반 자체 검색기.
- 검증된 지식 → 검색 → 충분히 학습된 자체 Transformer → fallback 순서.
- 자체 Transformer: RMSNorm + RoPE + SDPA causal attention + SwiGLU + weight tying.
- 답변 영역 SFT와 올바른 causal next-token shift 적용.
- random seed 기본 고정 제거. `--seed`를 지정할 때만 재현 생성.
- checkpoint 품질 gate: 충분히 학습되기 전에는 기본 hybrid가 신경망의 무근거 출력을 채택하지 않음.
- 최근 대화 history와 간단한 기억/계산 기능 포함.

## 모델 프리셋

| preset | context | hidden | heads | layers | 용도 |
|---|---:|---:|---:|---:|---|
| `main` | 256 | 224 | 7 | 5 | 기본 실사용/학습 |
| `large` | 512 | 320 | 8 | 8 | 더 큰 실험/충분한 연산 자원 |

vocab 크기는 데이터에 따라 달라진다. 현재 데이터에서 `main`은 약 4.5M parameters다.

## 바로 실행

```bash
pip install -r requirements.txt
python run_hgai.py --preset main
```

한 번만 질문:

```bash
python run_hgai.py --once "Casualties: Unknown 게임이 뭐야?"
```

실사용 기본은 `hybrid`다.

```bash
python run_hgai.py --mode hybrid
```

`raw-neural`은 검색/지식 안전망을 끄는 모델 진단용이다. 품질 평가 외에는 기본 사용을 권장하지 않는다.

## 데이터 생성

```bash
python build_dataset.py --max-variants 12
```

`train.txt`, `train_v10_extra.txt`, curated facts를 합치고 의미를 유지하는 자연어 변형을 만든다.

## 학습

새로 학습:

```bash
python run_hgai.py --retrain --preset main --steps 3000 --threads 2
```

이어 학습:

```bash
python run_hgai.py --resume --preset main --steps 1000 --threads 2
```

`main` 체크포인트는 기본 hybrid에서 다음 두 조건을 모두 만족해야 신경망 생성이 자동 활성화된다.

- 누적 step >= 1500
- best validation loss <= 2.8

그 전에도 검색/지식/계산 엔진은 정상 사용 가능하며 모델이 헛소리하면 사용자에게 노출되지 않는다.

## 평가

```bash
python evaluate_v10.py --mode search
python evaluate_v10.py --mode hybrid --model-file hgai_model_v10.pth
```

포함된 30문항 실사용 회귀 테스트는 search/hybrid 경로에서 30/30을 통과하도록 검증했다.

## 외부 모델을 참고만 하기

선택 사항이다.

```bash
pip install transformers torch accelerate
python teacher_reference.py --model Qwen/Qwen3-0.6B --prompts prompts.txt --output teacher_suggestions.jsonl
```

이 파일은 `reviewed:false`인 검토 큐다. HGAI가 이 모델을 실행 엔진으로 사용하지 않고 생성된 제안도 자동으로 학습 데이터에 섞지 않는다. 사람이 확인한 데이터만 별도로 옮겨야 한다.
