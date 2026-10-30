import os
import io
import logging
import uvicorn
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from PyPDF2 import PdfReader

# Langchain 컴포넌트 (최신 버전 호환 및 Chroma DB, MultiQuery 적용)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.docstore.document import Document

# 로컬 개발 환경용 변수 로드
load_dotenv()

# 환경 변수에서 구글 AI API 키 로드 (Supabase 관련 내용 제거)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("서버 초기화 치명적 실패! GOOGLE_API_KEY 변수가 누락되었습니다.")

# 다중 질문 생성 과정을 콘솔/로그에서 확인하기 위한 로깅 설정
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

try:
    # 1. 구글 임베딩 모델 설정
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    # 2. Chroma DB 초기화 (로컬 디렉토리 저장 방식)
    vector_db = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
except Exception as e:
    raise RuntimeError(f"Chroma DB 또는 임베딩 객체 초기화 중 시스템 에러 발생. 상세 원인: {str(e)}")

app = FastAPI(title="서울시 특화 도시 정책 RAG API 서버")

# 웹 클라이언트(GAS 등) 통신을 위한 CORS 미들웨어 적용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """입력한 PDF 파일을 텍스트를 변환하고 청크 단위로 쪼개기 및 Chroma DB 적재"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="요청 거부됨: 확장자가 유효하지 않습니다. '.pdf' 파일을 업로드해야 합니다.")
    
    try:
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        full_text = ""
        
        # PDF 각 페이지 순회하며 텍스트 변환 추출
        for page_num, page in enumerate(pdf_reader.pages):
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
                
        if not full_text.strip():
