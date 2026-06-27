# HGAI v6.2 Hybrid-Token

v6 기반에서 계산기 오작동, 키워드 섞임, GitHub Actions artifact 중첩 ZIP 문제를 고친 버전이다냥.

## 핵심 변경

- `train.txt`: 11,756개 Q&A
- 고유 질문: 약 10,312개 원본 / 정규화 기준 9천+개
- 계산기 판별 강화: `ㆍㆍ`, `ㅏㅡ`, `뭔소리야`, `계산하지마`가 계산기로 가지 않음
- 강한 키워드 보호: `LLM`, `GitHub`, `HG Company`, `퍼리`, `RAM`, `Python` 등이 서로 섞이지 않도록 우선 답변
- 지식/명령어/오류 대응/감정 대화 추가
- GitHub Actions artifact가 ZIP 안 ZIP이 아니라 바로 실행 파일 묶음으로 다운로드됨

## 폰에서 실행

```bash
cd ~/storage/downloads
unzip -o hgai-v6-2-model-mid-safe.zip -d hgai_v6_2_run
cd hgai_v6_2_run
python run_hgai.py --preset mid-safe --generation-mode safe
```

## 테스트

```bash
python run_hgai.py --preset mid-safe --generation-mode safe --benchmark
python run_hgai.py --preset mid-safe --once "LLM이 뭐야"
python run_hgai.py --preset mid-safe --once "HG Company 회사에 대해"
python run_hgai.py --preset mid-safe --once "계산하지마"
python run_hgai.py --preset mid-safe --once "10000×10000"
```

## GitHub Actions 학습

저장소에 push하면 자동 학습된다냥. 수동으로 돌릴 때는 Actions → Train HGAI v6.2 → Run workflow.

추천값:

- preset: `mid-safe`
- steps: `6000` 또는 `10000`
- generation_mode: `safe`

## 로컬 학습

```bash
python run_hgai.py --retrain --preset phone --steps 800 --threads 2
python run_hgai.py --retrain --preset mid-safe --steps 6000 --threads 2
```

## 모드

- `--generation-mode safe`: 검색/룰/보호 답변 우선, 필요하면 Torch 생성
- `--generation-mode creative`: 생성 답변을 더 적극 사용하지만 보호 키워드는 먼저 처리
- `--lite`: Torch 없이 검색/룰만 사용
