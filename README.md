# HGAI v7 Hybrid-Core

v7은 v6.2 기반에서 데이터와 실행 구조를 더 정리한 버전이다냥.

## 핵심 변화

- train.txt Q&A: 12,415개
- knowledge.json 강화: 지식/세계관/보호 키워드 분리
- personality.json 추가: 말투와 안전 규칙 분리
- eval_tests.json 추가: Actions에서 기본 성능 검사 가능
- `--learn "질문=답변"` 추가: 새 Q&A 한 줄 추가
- `--eval` 추가: 필수 테스트 자동 확인
- v6.2보다 오타 질문 보호 강화: `깃허브가 뮈야`, `LLM이 뭐샤` 등

## 폰 실행

```bash
cd ~/storage/downloads
unzip -o hgai_v7_hybrid_core.zip
cd hgai_v7_hybrid_core
python run_hgai.py --benchmark --lite
```

## Torch 모델 실행

```bash
python run_hgai.py --preset mid-safe --generation-mode safe
python run_hgai.py --preset mid-safe --generation-mode creative
```

## 학습

```bash
python run_hgai.py --retrain --preset mid-safe --steps 6000 --threads 2
```

## 평가

```bash
python run_hgai.py --eval --preset mid-safe --generation-mode safe
```

## 새 문장 추가

```bash
python run_hgai.py --learn "새 질문=새 답변이다냥"
```

## GitHub Actions

저장소에 push하면 `Train HGAI v7` 워크플로가 실행되고 artifact로 `hgai-v7-model-프리셋`이 생성된다냥.
