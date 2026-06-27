# HGAI v8 Hybrid-Engine

v8은 Transformer 앞에 6개 엔진을 붙인 버전이다냥.

## 추가 기능

1. 문장형 계산기: 사탕/남은 개수 같은 간단한 문장 문제 처리
2. 지식 DB 확장: ChatGPT, 카카오톡, 디스코드, 마이크로소프트, 리눅스 만든 사람 등
3. 한국어 수식 계산: 일 더하기 일, 백 더하기 오십, 십만 곱하기 이
4. 코드 생성기: 간단한 파이썬 코드 예시 출력
5. 기억 기능 강화: 이름과 좋아하는 것 저장
6. 자동 평가 강화: eval_tests.json 기반으로 GitHub Actions에서 품질 검사

## 실행

```bash
python run_hgai.py --preset mid-safe --generation-mode safe
```

## 평가

```bash
python run_hgai.py --eval --preset mid-safe --generation-mode safe
```

## 학습

```bash
python run_hgai.py --retrain --preset mid-safe --steps 10000 --threads 2
```

## GitHub Actions

push하면 Train HGAI v8 워크플로가 자동 실행된다냥.
