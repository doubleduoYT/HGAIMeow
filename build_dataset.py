#!/usr/bin/env python3
"""HGAI v10 dataset builder.

The final runtime is independent of external LLMs. This module expands the
hand-written HGAI data with conservative paraphrases and a curated factual
core. Optional teacher-model suggestions are kept separate and are never
required by HGAI at runtime.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CURATED_FACTS = {
    "Casualties: Unknown": "Casualties: Unknown은 수인을 잔혹하게 이용한 세계관의 하드코어 생존 게임이다냥 현재 덥듀가 모바일로 포팅 중이기도 하다냥",
    "HGAI": "HGAI는 HG Company에서 덥듀가 개발하는 한국어 고양이 말투 AI다냥",
    "인공지능": "인공지능은 컴퓨터가 학습, 추론, 인식 같은 지능적인 작업을 수행하도록 만드는 기술 분야다냥",
    "머신러닝": "머신러닝은 데이터에서 규칙과 패턴을 학습해 예측이나 판단을 하게 만드는 인공지능 방법이다냥",
    "딥러닝": "딥러닝은 여러 층의 신경망을 이용해 복잡한 패턴을 학습하는 머신러닝 방법이다냥",
    "신경망": "신경망은 입력을 여러 층의 수학적 변환에 통과시켜 패턴을 학습하는 모델 구조다냥",
    "LLM": "LLM은 아주 많은 텍스트로 학습해 다음 토큰을 예측하며 글을 이해하고 생성하는 대형 언어 모델이다냥",
    "Transformer": "Transformer는 어텐션을 이용해 토큰 사이 관계를 처리하는 신경망 구조다냥 현대 언어 모델에 널리 쓰인다냥",
    "어텐션": "어텐션은 현재 토큰을 처리할 때 다른 토큰 중 무엇을 얼마나 참고할지 계산하는 방식이다냥",
    "토큰": "토큰은 언어 모델이 문장을 처리할 때 사용하는 텍스트의 작은 단위다냥 단어 전체일 수도 있고 단어 조각이나 문자일 수도 있다냥",
    "파라미터": "파라미터는 신경망이 학습하면서 조정하는 숫자 값이다냥 모델의 지식을 담는 가중치가 대표적이다냥",
    "컨텍스트 길이": "컨텍스트 길이는 언어 모델이 한 번에 참고할 수 있는 토큰 수의 한도다냥",
    "temperature": "temperature는 생성 확률 분포의 날카로움을 조절하는 값이다냥 낮으면 보수적이고 높으면 다양해지는 경향이 있다냥",
    "top-p": "top-p는 누적 확률이 지정 값에 도달할 때까지의 후보 토큰만 남겨 샘플링하는 방식이다냥",
    "top-k": "top-k는 확률이 높은 상위 k개 토큰만 후보로 남겨 생성하는 방식이다냥",
    "양자화": "양자화는 모델 가중치나 활성값을 더 적은 비트로 표현해 메모리 사용량과 연산량을 줄이는 기법이다냥",
    "BitNet": "BitNet은 매우 낮은 비트의 가중치를 사용하도록 설계된 신경망 계열이다냥 단순히 일반 모델을 반올림하는 것과는 다르게 저비트 학습을 고려한다냥",
    "RAG": "RAG는 질문과 관련된 외부 자료를 먼저 검색한 뒤 그 내용을 참고해 답하게 만드는 방식이다냥",
    "SFT": "SFT는 질문과 좋은 답변 예시를 이용해 사전학습 모델을 지시 따르기 형태로 미세조정하는 학습 방식이다냥",
    "LoRA": "LoRA는 큰 모델의 모든 가중치를 바꾸지 않고 작은 저랭크 행렬을 학습해 효율적으로 미세조정하는 방법이다냥",
    "지식 증류": "지식 증류는 큰 교사 모델의 출력이나 분포를 참고해 더 작은 학생 모델을 학습시키는 방법이다냥",
    "오버피팅": "오버피팅은 학습 데이터에는 지나치게 잘 맞지만 새로운 데이터에는 성능이 떨어지는 현상이다냥",
    "검증 데이터": "검증 데이터는 학습에 직접 쓰지 않고 학습 중 일반화 성능을 확인하는 데 사용하는 데이터다냥",
    "손실 함수": "손실 함수는 모델의 예측이 정답과 얼마나 다른지 숫자로 나타내 학습 방향을 정하는 함수다냥",
    "gradient": "gradient는 손실을 줄이기 위해 각 파라미터를 어느 방향으로 얼마나 바꿔야 하는지 나타내는 미분 값이다냥",
    "AdamW": "AdamW는 적응적 학습률과 분리된 weight decay를 사용하는 신경망 최적화 알고리즘이다냥",
    "Python": "Python은 읽기 쉬운 문법과 풍부한 라이브러리를 가진 범용 프로그래밍 언어다냥",
    "C": "C는 시스템 프로그래밍과 임베디드 분야에서 널리 쓰이는 컴파일 언어다냥 메모리를 비교적 직접 다룰 수 있다냥",
    "C++": "C++는 C 계열의 고성능 범용 언어로 객체지향과 템플릿 같은 기능을 제공한다냥",
    "Rust": "Rust는 메모리 안전성과 성능을 함께 목표로 하는 시스템 프로그래밍 언어다냥",
    "Java": "Java는 JVM 위에서 실행되는 범용 객체지향 언어로 서버와 안드로이드 분야에서 많이 쓰여 왔다냥",
    "JavaScript": "JavaScript는 웹 브라우저에서 동적인 기능을 구현하고 서버에서도 사용할 수 있는 프로그래밍 언어다냥",
    "Git": "Git은 파일 변경 이력을 저장하고 브랜치와 병합으로 협업을 돕는 분산 버전 관리 시스템이다냥",
    "GitHub": "GitHub는 Git 저장소를 호스팅하고 이슈, Pull Request, Actions 같은 협업 기능을 제공하는 서비스다냥",
    "GitHub Actions": "GitHub Actions는 저장소 이벤트에 맞춰 빌드, 테스트, 배포 같은 작업을 자동으로 실행하는 CI/CD 기능이다냥",
    "Linux": "Linux는 리누스 토르발스가 시작한 오픈소스 커널이며 여러 배포판의 기반으로 사용된다냥",
    "Windows": "Windows는 마이크로소프트가 개발하는 데스크톱과 서버용 운영체제 계열이다냥",
    "Android": "Android는 리눅스 커널을 기반으로 하는 모바일 운영체제 플랫폼이다냥",
    "Termux": "Termux는 안드로이드에서 터미널과 리눅스 계열 사용자 공간 도구를 사용할 수 있게 해주는 앱이다냥",
    "ADB": "ADB는 Android Debug Bridge의 약자로 컴퓨터에서 안드로이드 기기와 통신해 디버깅, 파일 전송, 명령 실행을 할 수 있는 도구다냥",
    "APK": "APK는 안드로이드 앱을 설치하고 배포할 때 사용하는 패키지 형식이다냥",
    "NDK": "Android NDK는 C와 C++ 같은 네이티브 코드를 안드로이드 앱에서 빌드하고 사용할 수 있게 해주는 도구 모음이다냥",
    "Unity": "Unity는 게임과 실시간 2D·3D 콘텐츠를 만드는 게임 엔진이다냥 C# 스크립팅을 주로 사용한다냥",
    "Godot": "Godot은 오픈소스 게임 엔진으로 2D와 3D 게임 제작을 지원하고 GDScript, C# 등을 사용할 수 있다냥",
    "Pygame": "Pygame은 Python에서 창, 입력, 이미지, 소리 같은 2D 게임 기능을 다루게 해주는 라이브러리다냥",
    "CPU": "CPU는 범용 명령을 실행하며 운영체제와 프로그램의 주요 연산을 처리하는 프로세서다냥",
    "GPU": "GPU는 많은 연산을 병렬로 처리하는 데 강한 프로세서로 그래픽과 인공지능 계산에 널리 쓰인다냥",
    "RAM": "RAM은 실행 중인 프로그램과 데이터를 빠르게 임시 저장하는 휘발성 메모리다냥",
    "SSD": "SSD는 플래시 메모리를 이용해 데이터를 저장하는 저장장치로 일반적으로 HDD보다 빠르다냥",
    "TCP": "TCP는 연결을 만들고 순서와 재전송을 관리해 신뢰성 있는 바이트 스트림을 제공하는 전송 계층 프로토콜이다냥",
    "UDP": "UDP는 연결 설정과 재전송을 기본 제공하지 않는 가벼운 데이터그램 전송 프로토콜이다냥 지연이 중요한 용도에 자주 쓰인다냥",
    "TCP와 UDP 차이": "TCP는 연결과 순서, 재전송을 관리해 신뢰성을 높이고 UDP는 그런 보장을 줄여 오버헤드와 지연을 낮춘다냥",
    "IP": "IP는 네트워크에서 패킷의 출발지와 목적지를 주소로 식별해 전달하는 인터넷 계층 프로토콜이다냥",
    "DNS": "DNS는 도메인 이름을 IP 주소 같은 네트워크 정보로 변환해주는 분산 이름 시스템이다냥",
    "HTTP": "HTTP는 웹에서 클라이언트와 서버가 요청과 응답을 주고받는 응용 계층 프로토콜이다냥",
    "HTTPS": "HTTPS는 HTTP 통신을 TLS로 암호화해 도청과 변조 위험을 줄이는 방식이다냥",
    "SSH": "SSH는 원격 시스템에 암호화된 연결로 로그인하거나 명령을 실행할 때 사용하는 프로토콜이다냥",
    "API": "API는 프로그램끼리 기능이나 데이터를 정해진 규칙으로 주고받게 하는 인터페이스다냥",
    "REST": "REST는 자원을 URI로 표현하고 HTTP 메서드를 활용하는 웹 API 설계 방식이다냥",
    "JSON": "JSON은 객체와 배열 중심의 간단한 텍스트 데이터 교환 형식이다냥",
    "YAML": "YAML은 사람이 읽기 쉬운 들여쓰기 기반 데이터 직렬화 형식으로 설정 파일에 자주 쓰인다냥",
    "XML": "XML은 태그로 계층 구조 데이터를 표현하는 텍스트 형식이다냥",
    "HTML": "HTML은 웹 문서의 구조와 의미를 표현하는 마크업 언어다냥",
    "CSS": "CSS는 HTML 문서의 색상, 배치, 크기 같은 표현 방식을 지정하는 스타일 언어다냥",
    "SQL": "SQL은 관계형 데이터베이스에서 데이터를 조회하고 수정하고 구조를 관리하는 언어다냥",
    "SQLite": "SQLite는 별도 서버 없이 하나의 파일을 중심으로 동작하는 경량 관계형 데이터베이스 엔진이다냥",
    "프로세스": "프로세스는 실행 중인 프로그램의 인스턴스로 독립적인 주소 공간과 자원을 가진다냥",
    "스레드": "스레드는 한 프로세스 안에서 실행 흐름을 나누는 단위로 같은 프로세스의 메모리를 공유한다냥",
    "멀티스레딩": "멀티스레딩은 한 프로세스에서 여러 스레드를 사용해 작업을 동시에 또는 겹쳐 처리하는 방식이다냥",
    "컴파일": "컴파일은 소스 코드를 다른 형태의 실행 코드나 중간 코드로 변환하는 과정이다냥",
    "인터프리터": "인터프리터는 프로그램을 실행할 때 코드를 해석하며 동작시키는 실행 방식 또는 프로그램이다냥",
    "버그": "버그는 프로그램이 의도와 다르게 동작하게 만드는 결함이나 오류다냥",
    "디버깅": "디버깅은 프로그램 오류의 원인을 찾아 재현하고 수정하는 과정이다냥",
    "테스트": "소프트웨어 테스트는 프로그램이 기대한 대로 동작하는지 자동 또는 수동으로 확인하는 과정이다냥",
    "유닛 테스트": "유닛 테스트는 함수나 클래스 같은 작은 단위를 독립적으로 검증하는 테스트다냥",
    "CI": "CI는 코드 변경을 자주 통합하고 자동 빌드와 테스트로 문제를 빠르게 찾는 개발 방식이다냥",
    "CD": "CD는 검증된 소프트웨어를 자동으로 배포하거나 배포 가능한 상태로 유지하는 개발 방식이다냥",
    "Docker": "Docker는 애플리케이션과 의존성을 컨테이너 이미지로 묶어 일관된 환경에서 실행하게 하는 도구다냥",
    "컨테이너": "컨테이너는 호스트 커널을 공유하면서 프로세스와 파일 시스템 등을 격리해 실행하는 환경이다냥",
    "가상 머신": "가상 머신은 하이퍼바이저 위에서 독립된 운영체제를 실행하는 가상 컴퓨터다냥",
    "암호화": "암호화는 허가되지 않은 사람이 내용을 읽기 어렵도록 데이터를 키를 이용해 변환하는 기술이다냥",
    "해시": "해시는 임의 길이 입력을 고정 길이 값으로 변환하는 함수다냥 암호학적 해시는 역산과 충돌을 어렵게 설계한다냥",
    "SHA-256": "SHA-256은 256비트 해시 값을 만드는 암호학적 해시 함수다냥",
    "base64": "Base64는 이진 데이터를 ASCII 문자로 표현하는 인코딩 방식이지 암호화 방식은 아니다냥",
    "Vulkan": "Vulkan은 GPU 그래픽과 계산을 낮은 수준에서 제어할 수 있게 설계된 크로스 플랫폼 그래픽 API다냥",
    "OpenGL": "OpenGL은 2D와 3D 그래픽 렌더링을 위한 크로스 플랫폼 그래픽 API다냥",
    "프레임률": "프레임률은 1초에 화면을 몇 번 갱신하는지 나타내며 보통 FPS로 표현한다냥",
    "지연시간": "지연시간은 요청이나 입력이 결과로 반영되기까지 걸리는 시간이다냥",
    "FPS 드랍": "FPS 드랍은 CPU나 GPU 병목, 과도한 할당과 GC, I/O, 발열 제한 등 여러 원인으로 프레임 처리 시간이 늘어날 때 생길 수 있다냥",
    "프레임 드랍": "프레임 드랍은 CPU나 GPU 병목, 과도한 할당과 GC, I/O, 발열 제한 등 여러 원인으로 프레임 처리 시간이 늘어날 때 생길 수 있다냥",
    "메모리 누수": "메모리 누수는 더 이상 필요하지 않은 메모리를 해제하거나 회수하지 못해 사용량이 계속 늘어나는 문제다냥",
    "가비지 컬렉션": "가비지 컬렉션은 더 이상 접근할 수 없는 객체의 메모리를 자동으로 회수하는 관리 방식이다냥",
    "자료구조": "자료구조는 데이터를 저장하고 접근하고 수정하기 위한 조직 방식이다냥 배열, 리스트, 해시 테이블 등이 있다냥",
    "알고리즘": "알고리즘은 문제를 해결하기 위한 명확한 절차와 규칙의 집합이다냥",
    "시간 복잡도": "시간 복잡도는 입력 크기가 커질 때 알고리즘 실행 시간이 어떻게 증가하는지 나타내는 척도다냥",
    "Big-O": "Big-O 표기법은 입력 크기에 따른 시간이나 공간 사용 증가율의 상한을 표현하는 방법이다냥",
    "배열": "배열은 같은 종류의 여러 값을 인덱스로 접근할 수 있게 연속적으로 다루는 자료구조다냥",
    "스택": "스택은 마지막에 넣은 값을 먼저 꺼내는 LIFO 방식의 자료구조다냥",
    "큐": "큐는 먼저 넣은 값을 먼저 꺼내는 FIFO 방식의 자료구조다냥",
    "해시 테이블": "해시 테이블은 키를 해시 값으로 변환해 값을 빠르게 찾도록 만든 자료구조다냥",
    "재귀": "재귀는 함수가 문제를 더 작은 같은 형태의 문제로 나누며 자기 자신을 호출하는 기법이다냥",
    "정규표현식": "정규표현식은 문자열의 패턴을 표현해 검색, 검사, 치환에 사용하는 문법이다냥",
    "Unicode": "Unicode는 세계 여러 문자에 고유한 코드 포인트를 부여하는 문자 표준이다냥",
    "UTF-8": "UTF-8은 Unicode 코드 포인트를 1~4바이트로 표현하는 가변 길이 문자 인코딩이다냥",
    "MIDI": "MIDI는 실제 음원을 저장하기보다 음표, 악기, 세기, 타이밍 같은 연주 이벤트를 표현하는 규격이다냥",
    "WAV": "WAV는 주로 비압축 PCM 오디오를 담는 데 쓰이는 컨테이너 형식이다냥",
    "MP3": "MP3는 사람이 덜 민감한 소리 정보를 줄여 용량을 낮추는 손실 압축 오디오 형식이다냥",
}

QUESTION_PATTERNS = ["{term}이 뭐야?", "{term}가 뭐야?", "{term}는 뭐야?", "{term}란 뭐야?", "{term} 설명해줘", "{term} 쉽게 설명해줘", "{term} 뜻이 뭐야?", "{term} 알려줘"]
SPECIAL_QUESTIONS = {
    "Casualties: Unknown": ["Casualties: Unknown 게임이 뭐야?", "Casualties: Unknown이 뭐야?", "Casualties Unknown 게임 설명해줘", "캐주얼티즈 언노운이 뭐야?", "Casualties: Unknown 설명해줘", "Casualties Unknown은 어떤 게임이야?"],
    "TCP와 UDP 차이": ["TCP랑 UDP는 뭐가 달라?", "TCP와 UDP 차이 알려줘", "TCP UDP 차이가 뭐야?", "TCP랑 UDP 비교해줘"],
}
GENERIC_REWRITES = [
    (re.compile(r"(.+?)(?:이|가|는|은) 뭐야\??$"), ["{x} 설명해줘", "{x}가 무슨 뜻이야?", "{x} 쉽게 알려줘", "{x}에 대해 알려줘"]),
    (re.compile(r"(.+?) 뜻이 뭐야\??$"), ["{x}가 무슨 뜻이야?", "{x} 설명해줘", "{x}는 어떤 뜻이야?"]),
    (re.compile(r"(.+?) 설명해줘\??$"), ["{x} 쉽게 설명해줘", "{x} 알려줘", "{x}가 뭐야?"]),
    (re.compile(r"(.+?) 알려줘\??$"), ["{x} 설명해줘", "{x}에 대해 알려줘", "{x}가 뭐야?"]),
]

def clean(s: str) -> str: return re.sub(r"\s+", " ", str(s).strip())
def parse_pairs(text: str):
    out=[]
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        q,a=line.split("=",1); q,a=clean(q),clean(a)
        if q and a: out.append((q,a))
    return out

def curated_pairs():
    out=[]
    for term,answer in CURATED_FACTS.items():
        qs=list(SPECIAL_QUESTIONS.get(term,[]))+[p.format(term=term) for p in QUESTION_PATTERNS]
        for q in qs: out.append((clean(q),clean(answer)))
    return out

def paraphrase_question(q: str):
    q=clean(q); variants={q,q.rstrip("?!.")}; variants.add(q.replace("알려 줘","알려줘").replace("설명 해줘","설명해줘")); bare=q.rstrip("?!.")
    if 2<=len(bare)<=120:
        variants.update({f"궁금한데 {q}",f"하나 물어볼게 {q}",f"질문 하나 할게 {q}",f"혹시 {q}",f"{bare} 좀 알려줘",f"{bare}에 대해 답해줘"})
    for rx,templates in GENERIC_REWRITES:
        m=rx.fullmatch(q)
        if m:
            x=clean(m.group(1))
            if 1<len(x)<=80: variants.update(t.format(x=x) for t in templates)
    if q.endswith("?"): variants.add(q[:-1])
    if "뭐야" in q: variants.add(q.replace("뭐야","뭐임")); variants.add(q.replace("뭐야","무엇이야"))
    if "알려줘" in q: variants.add(q.replace("알려줘","알려 줄래?"))
    return [clean(v) for v in variants if clean(v)]

def is_obsolete_pair(q: str, a: str) -> bool:
    text=(clean(q)+" "+clean(a)).lower()
    return "phone preset" in text or ("phone은 폰용" in text and "mid-safe" in text)

def expand_pairs(base_pairs,max_variants=8):
    source=[(q,a) for q,a in base_pairs if not is_obsolete_pair(q,a)]+curated_pairs(); out=[]; seen=set()
    for q,a in source:
        for v in paraphrase_question(q)[:max_variants]:
            key=(v,a)
            if key not in seen: seen.add(key); out.append(key)
    return out

def dataset_hash(pairs): return hashlib.sha256("\n".join(f"{q}={a}" for q,a in pairs).encode("utf-8")).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train",default=str(BASE_DIR/"train.txt")); ap.add_argument("--output",default=str(BASE_DIR/"augmented_train.txt")); ap.add_argument("--report",default=str(BASE_DIR/"DATASET_REPORT_V10.json")); ap.add_argument("--max-variants",type=int,default=8); args=ap.parse_args()
    train_path=Path(args.train); base=parse_pairs(train_path.read_text(encoding="utf-8")); extra_path=train_path.with_name("train_v10_extra.txt")
    if extra_path.exists(): base+=parse_pairs(extra_path.read_text(encoding="utf-8"))
    pairs=expand_pairs(base,args.max_variants); Path(args.output).write_text("\n".join(f"{q}={a}" for q,a in pairs)+"\n",encoding="utf-8")
    report={"version":"v10","base_pairs":len(base),"curated_topics":len(CURATED_FACTS),"effective_pairs":len(pairs),"unique_questions":len({q for q,_ in pairs}),"sha256":dataset_hash(pairs),"external_model_required_at_runtime":False}
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
