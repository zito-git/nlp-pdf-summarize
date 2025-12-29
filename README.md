# NLP PDF QA API (FastAPI + FAISS + Gemini + GPU)

이 프로젝트는 PDF 문서를 업로드하고, 그 내용을 기반으로 질문에 답변하는 **FastAPI 애플리케이션**입니다.  
SentenceTransformer로 PDF 내용을 임베딩하고, **FAISS GPU 인덱스**로 검색하며,  
**Google Gemini API**를 통해 자연스러운 답변을 생성합니다.

---

## 주요 기술 스택

- **FastAPI** — RESTful API 서버
- **SentenceTransformer** — 한국어 임베딩 (`jhgan/ko-sroberta-multitask`)
- **FAISS GPU** — 벡터 검색
- **Google Gemini API** — 답변 생성
- **python-dotenv** — 환경 변수 관리
- **Docker Compose** — GPU 실행 환경

---

## 프로젝트 구조

```bash
project/
├── main.py                 # FastAPI 메인 코드
├── requirements.txt        # 패키지 목록
├── docker-compose.yml      # Docker 구성
├── Dockerfile              # Docker 빌드 설정
├── .env                    # 환경 변수 파일 (API 키 등)
├── data/                   # PDF 임베딩 및 인덱스 저장
└── README.md
```

---

## 환경 변수 설정 (.env)

```bash
#  루트 디렉토리에 .env 파일을 생성
GENAI_KEY=YOUR_GEMINI_API_KEY
```

---

## Docker 환경 구성

```bash
# --------------------------
#  Dockerfile
# --------------------------
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 소스 복사
COPY main.py .

# FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

```

---

## docker-compose.yml

```bash
version: "3.9"

services:
  npl-fastapi:
    build: .
    container_name: npl-fastapi
    restart: always
    runtime: nvidia
    env_file:
      - .env
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data

```

---

## requirements.txt

```bash
fastapi==0.118.3
uvicorn==0.23.2
python-multipart==0.0.6
pypdf==3.17.0
sentence-transformers==2.2.2
faiss-gpu==1.7.2
huggingface-hub
transformers==4.35.0
numpy==1.25.2
requests==2.32.0
google-generativeai
torch
python-dotenv==1.0.1
```

---

## 실행 방법

1. NVIDIA Container Toolkit 설치
   (GPU 사용 시)

```bash
sudo apt remove -y nvidia-docker2
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. Docker Compose 실행

```bash
docker compose up --build -d
```

3. 서버 접속
   Swagger UI: http://localhost:8000/docs

---

## API 엔드포인트

#### /upload — PDF 업로드

**Method:** POST  
**Form-Data:**

```bash
file: sample.pdf
```

Response:

```json
{
  "message": "PDF 업로드 완료",
  "chunks": 128,
  "session_id": "e1b2c3d4-5678-90ab-cdef-1234567890ab"
}
```

---

#### /ask — 질문하기

**Method:** POST  
**Form-Data:**

```
question: 질문 내용
session_id: 업로드 시 받은 세션 UUID
```

Response:

```json
{
  "request_uuid": "d4a8e8b0-92e3-4f6f-8bb0-6bcd5cbd9a50",
  "answer": "문서의 주요 내용은 ~입니다."
}
```

---

## 로컬에서 실행 (Docker 없이)

```bash
pip install -r requirements.txt
python main.py
```
