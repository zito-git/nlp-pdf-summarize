# 🧠 NLP PDF QA API (FastAPI + FAISS + Gemini + GPU)

이 프로젝트는 PDF 문서를 업로드하고, 그 내용을 기반으로 질문에 답변하는 **FastAPI 애플리케이션**입니다.  
SentenceTransformer로 PDF 내용을 임베딩하고, **FAISS GPU 인덱스**로 검색하며,  
**Google Gemini API**를 통해 자연스러운 답변을 생성합니다.

---

## ⚙️ 주요 기술 스택

- **FastAPI** — RESTful API 서버
- **SentenceTransformer** — 한국어 임베딩 (`jhgan/ko-sroberta-multitask`)
- **FAISS GPU** — 벡터 검색
- **Google Gemini API** — 답변 생성
- **python-dotenv** — 환경 변수 관리
- **Docker Compose** — GPU 실행 환경

---

## 📁 프로젝트 구조

```bash
project/
├── main.py                 # FastAPI 메인 코드
├── requirements.txt        # 패키지 목록
├── docker-compose.yml      # Docker 구성
├── Dockerfile              # Docker 빌드 설정
├── .env                    # 환경 변수 파일 (API 키 등)
├── data/                   # PDF 임베딩 및 인덱스 저장
└── README.md
