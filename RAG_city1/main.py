import os
import io
import logging
import uvicorn
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
 
from PyPDF2 import PdfReader
 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
 
load_dotenv()
 
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("서버 초기화 치명적 실패! GOOGLE_API_KEY 변수가 누락되었습니다.")
 
# ───────────────────────────────────────────────
# RAG 설정 (클라이언트 화면에 그대로 표시되는 값)
# ───────────────────────────────────────────────
SYSTEM_ROLE = "당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요."
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-3.5-flash-lite"
RETRIEVER_K = 5
 
RAG_CONFIG = {
    "vector_db": "Chroma DB (persist_directory=./chroma_db)",
    "embedding_model": EMBEDDING_MODEL,
    "llm_model": LLM_MODEL,
    "retriever": f"MultiQueryRetriever (LLM이 원 질의를 다각도 질의로 재생성 → 각각 top-{RETRIEVER_K} 검색 → 중복 제거 합집합)",
    "query_expansion": "대화 기록 기반 질의 재구성(Condense) + 검색 친화적 질문으로 구체화 (LLM, temperature=0)",
    "system_role": SYSTEM_ROLE,
    "chunking": "RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)",
}
 
# MultiQueryRetriever가 생성한 질의가 Cloud Run 로그에 찍히도록 두 로거 이름 모두 활성화
logging.basicConfig(level=logging.INFO)
for name in ("langchain.retrievers.multi_query", "langchain_classic.retrievers.multi_query"):
    logging.getLogger(name).setLevel(logging.INFO)
logger = logging.getLogger("seoul-rag")
 
try:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
except Exception as e:
    raise RuntimeError(f"Chroma DB 또는 임베딩 객체 초기화 중 시스템 에러 발생. 상세 원인: {str(e)}")
 
# LLM은 한 번만 생성해서 재사용
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0)
answer_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2)
 
app = FastAPI(title="서울시 특화 도시 정책 RAG API 서버")
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
 
 
# ───────────────────────────────────────────────
# 상태/설정 엔드포인트 (클라이언트 '연결 상태' 및 'RAG 설정' 표시용)
# ───────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Chroma DB + MultiQueryRetriever 기반 서울시 도시계획 챗봇 서버가 정상 작동 중입니다."}
 
 
@app.get("/health")
def health():
    try:
        count = vector_db._collection.count()
    except Exception:
        count = -1
    return {"status": "ok", "indexed_chunks": count}
 
 
@app.get("/config")
def get_config():
    return RAG_CONFIG
 
 
# ───────────────────────────────────────────────
# PDF 업로드 → 청크 분할 → Chroma DB 적재
# ───────────────────────────────────────────────
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="요청 거부됨: '.pdf' 파일을 업로드해야 합니다.")
 
    try:
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
 
        documents: List[Document] = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        # 페이지 번호를 metadata에 남겨 답변 출처 표시에 사용
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            extracted = page.extract_text()
            if not extracted:
                continue
            for chunk in text_splitter.split_text(extracted):
                documents.append(Document(page_content=chunk, metadata={"source": file.filename, "page": page_num}))
 
        if not documents:
            raise HTTPException(status_code=422, detail="문서 분석 실패: 추출 가능한 텍스트를 찾지 못했습니다.")
 
        try:
            vector_db.add_documents(documents)
        except Exception as db_err:
            raise HTTPException(status_code=500, detail=f"Chroma DB 데이터 적재 실패: {str(db_err)}")
 
        return {
            "status": "success",
            "filename": file.filename,
            "pages": len(pdf_reader.pages),
            "chunks": len(documents),
            "message": f"총 {len(documents)}개의 청크가 Chroma DB에 적재되었습니다.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"알 수 없는 서버 오류: {str(e)}")
 
 
# ───────────────────────────────────────────────
# 질의 확장 → 다중 질문 검색 → 시스템 역할 기반 답변 (JSON 응답)
# ───────────────────────────────────────────────
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    history_text = "\n".join(
        f"{'사용자' if m.get('role') == 'user' else 'AI'}: {m.get('content')}" for m in request.chat_history
    )
 
    # 1) 질의 확장 — 대화 기록 유무와 관계없이 항상 수행
    try:
        condense_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "너는 검색 질의 재작성기다. 대화 기록을 참고해 사용자의 의도를 파악하고, "
             "대명사(이/그/저 등)를 명확한 명사로 바꾸며, 문서 검색에 유리하도록 구체적인 한 문장의 질문으로 다시 써라. "
             "대화 기록이 없으면 원 질문의 의미를 유지한 채 핵심 키워드가 드러나도록 다듬어라. 질문 한 문장만 출력하라."),
            ("human", "[대화 기록]\n{history_text}\n\n[현재 질문]\n{query}"),
        ])
        augmented_query = (condense_prompt | llm | StrOutputParser()).invoke(
            {"history_text": history_text or "(없음)", "query": request.query}
        ).strip()
        logger.info("질의 확장: '%s' -> '%s'", request.query, augmented_query)
    except Exception as aug_err:
        raise HTTPException(status_code=500, detail=f"질의 확장 실행 실패: {str(aug_err)}")
 
    # 2) MultiQueryRetriever — 생성된 질의 목록을 클라이언트에 그대로 전달
    try:
        base_retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})
        multi_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm, include_original=True)
 
        generated_queries: List[str] = multi_retriever.llm_chain.invoke({"question": augmented_query})
        generated_queries = [q.strip() for q in generated_queries if q and q.strip()]
        all_queries = generated_queries + [augmented_query]
        logger.info("MultiQuery 생성 질의: %s", generated_queries)
 
        docs_per_query = [base_retriever.invoke(q) for q in all_queries]
        retrieved_docs = multi_retriever.unique_union([d for docs in docs_per_query for d in docs])
 
        context_text = "\n\n".join(d.page_content for d in retrieved_docs) if retrieved_docs else "검색된 관련 문서가 존재하지 않습니다."
        sources = [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page"), "preview": d.page_content[:120]}
            for d in retrieved_docs
        ]
    except Exception as db_search_err:
        raise HTTPException(status_code=500, detail=f"Chroma DB 다중 질문 검색 실패: {str(db_search_err)}")
 
    # 3) 시스템 역할 정의가 포함된 최종 답변
    try:
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_ROLE + " 반드시 아래 [문서 정보]에 근거해 답하고, 문서에 없는 내용은 지어내지 마세요."),
            ("human", "[문서 정보]\n{context}\n\n[사용자 질문]\n{question}"),
        ])
        answer = (rag_prompt | answer_llm | StrOutputParser()).invoke(
            {"context": context_text, "question": augmented_query}
        )
    except Exception as rag_err:
        raise HTTPException(status_code=500, detail=f"최종 응답 파이프라인 구성 실패: {str(rag_err)}")
 
    return {
        "original_query": request.query,
        "augmented_query": augmented_query,
        "generated_queries": generated_queries,
        "retrieved_doc_count": len(retrieved_docs),
        "sources": sources,
        "answer": answer,
        "config": RAG_CONFIG,
    }
 
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
 
