# HGAI v6 Hybrid-Token

HGAI v6는 기존 `질문=답변` 방식은 유지하면서, 검색/룰/계산기/메모리/토큰 기반 Transformer를 합친 버전이야.

## 이번 패키지 상태

- train.txt: 10,731개 Q&A
- 고유 질문: 9,248개
- 기존 v5 데이터 유지/정리 + 퍼리/HG Company/덥듀/LLM/GitHub 관련 질문 확장
- `냥` 띄어쓰기 자동 보정
- GitHub Actions 자동 학습 포함

## 실행

```bash
python run_hgai.py --preset mid-safe
```

한 번만 테스트:

```bash
python run_hgai.py --once "퍼리가 뭐야"
python run_hgai.py --once "너는 어디에서 만들어졌어?"
python run_hgai.py --once "LLM이 뭐야"
```

Torch 없이:

```bash
python run_hgai.py --lite
```

## 학습

폰/Termux:

```bash
python run_hgai.py --retrain --preset phone --steps 800 --threads 2
```

GitHub Actions/PC:

```bash
python run_hgai.py --retrain --preset mid-safe --steps 6000 --threads 2
```

더 크게:

```bash
python run_hgai.py --retrain --preset mid-plus --steps 10000 --threads 2
```

## 모드

- `--generation-mode safe`: 기본값. 정확/계산/검색 우선, 필요하면 Transformer
- `--generation-mode creative --reply-mode torch-first`: Transformer 생성 먼저 시도
- `--generation-mode off`: 검색/룰만 사용
- `--lite`: Torch import 자체를 안 함

## 정보 확인

```bash
python run_hgai.py --info --preset mid-safe
python run_hgai.py --benchmark --preset mid-safe
```


## 참고

이 ZIP에는 일부러 `hgai_model.pth`를 넣지 않았어. v6는 토큰 구조가 바뀌어서 GitHub Actions나 PC에서 새로 학습해야 해.
