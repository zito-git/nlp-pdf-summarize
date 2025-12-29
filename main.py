from fastapi import FastAPI, File, UploadFile, Form
import uuid
import os
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pickle
from dotenv import load_dotenv 


app = FastAPI()

# -------------------------
# Gemini API 키
# -------------------------

load_dotenv()
GENAI_KEY = os.getenv("GENAI_KEY")
genai.configure(api_key=GENAI_KEY)

# -------------------------
# 임베딩 모델 (GPU로 자동)
# -------------------------
EMBED_MODEL = SentenceTransformer("jhgan/ko-sroberta-multitask", device="cuda")

# -------------------------
# 데이터 저장 경로
# -------------------------
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)


def save_index(uuid_str, index, chunks):
    faiss.write_index(
        faiss.index_gpu_to_cpu(index), os.path.join(DATA_DIR, f"{uuid_str}.index")
    )
    with open(os.path.join(DATA_DIR, f"{uuid_str}.pkl"), "wb") as f:
        pickle.dump(chunks, f)


def load_index(uuid_str):
    cpu_index = faiss.read_index(os.path.join(DATA_DIR, f"{uuid_str}.index"))
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    with open(os.path.join(DATA_DIR, f"{uuid_str}.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return gpu_index, chunks


# -------------------------
# 업로드
# -------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    pdf = PdfReader(file.file)
    text = "".join(page.extract_text() or "" for page in pdf.pages)

    chunks_local = [text[i : i + 500] for i in range(0, len(text), 500)]
    embeddings = np.array(EMBED_MODEL.encode(chunks_local))

    # FAISS GPU 인덱스
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(embeddings.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index.add(embeddings)

    # UUID 생성
    session_id = str(uuid.uuid4())
    save_index(session_id, gpu_index, chunks_local)

    return {
        "message": "PDF 업로드 완료",
        "chunks": len(chunks_local),
        "session_id": session_id,
    }


# -------------------------
# 질문
# -------------------------
@app.post("/ask")
async def ask_question(question: str = Form(...), session_id: str = Form(...)):
    try:
        gpu_index, chunks_local = load_index(session_id)
    except FileNotFoundError:
        return {"error": "세션 없음. 먼저 업로드하세요."}

    q_emb = np.array([EMBED_MODEL.encode(question)])
    D, I = gpu_index.search(q_emb, 3)
    context = "\n".join([chunks_local[i] for i in I[0]])

    prompt = f"""너는 PDF 문서를 기반으로 질문에 답변하는 AI야.
문서 내용: {context}
질문: {question}"""

    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)

    # 사용 후 GPU 메모리 해제
    del gpu_index

    return {"request_uuid": str(uuid.uuid4()), "answer": response.text}
