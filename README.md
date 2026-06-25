# HGAI v5 Clean

v4 train.txt를 직접 확인해서 질문-답변이 안 맞는 자동 생성 노이즈를 정리하고, 기존 말투와 데이터는 최대한 유지한 버전이야.

## 데이터 통계

- 기존 v4 pairs: 9,567
- 정리 후 유지한 기존 pairs: 8,597
- v5 최종 pairs: 9,345
- 고유 질문 수: 7,884
- 제거한 노이즈 추정: 970

## 추가 강화

- 퍼리/수인 관련 질문
- HGAI 출신, 덥듀, HG Company 관련 질문
- Python, C언어, 리눅스, 윈도우, Termux, GitHub, GitHub Actions
- 파라미터, 토큰, 에포크, 과적합, 손실값 등 AI 학습 용어
- 감정 위로, 공부, 코딩, 서버 관리, 매운맛/팩폭 대화

## 실행

```bash
python run_hgai.py --lite
```

## 폰에서 Torch 학습

```bash
python run_hgai.py --retrain --preset phone --steps 500 --threads 2
```

## GitHub Actions/PC 추천 학습

```bash
python run_hgai.py --retrain --preset mid-safe --steps 3000 --threads 2
```

## 테스트

```bash
python run_hgai.py --once "퍼리가 뭐야" --lite
python run_hgai.py --once "너는 어디에서 만들어졌어" --lite
python run_hgai.py --once "HG Company가 뭐야" --lite
```
