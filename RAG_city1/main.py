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
            raise HTTPException(status_code=422, detail="문서 분석 실패: 추출 가능한 텍스트를 찾지 못했습니다.")

        # 텍스트를 청크 단위로 쪼개기 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(full_text)
        
        # Chroma DB 적재를 위해 LangChain Document 객체로 변환
        documents = [Document(page_content=chunk, metadata={"source": file.filename}) for chunk in chunks]
        
        try:
            # 벡터 DB(Chroma DB)에 데이터 적재
            vector_db.add_documents(documents)
        except Exception as db_err:
            raise HTTPException(status_code=500, detail=f"Chroma DB 데이터 적재 실패: {str(db_err)}")
                
        return {"status": "success", "message": f"총 {len(chunks)}개의 청크 조각이 Chroma DB에 안전하게 적재되었습니다."}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"알 수 없는 서버 오류: {str(e)}")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """다중 질문 생성 및 질의 확장을 거친 후 Chroma DB 리트리버를 통해 관련 문서를 찾고 스트리밍 답변을 제공"""
    
    history_text = ""
    for msg in request.chat_history:
        role = "사용자" if msg.get("role") == "user" else "AI"
        history_text += f"{role}: {msg.get('content')}\n"

    try:
        # [질의 확장 단계] 대화 맥락을 포함한 질문 재구성
        condense_prompt = ChatPromptTemplate.from_template(
            "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라.\n\n"
            "[대화 기록]\n{history_text}\n\n"
            "[현재 질문]\n{query}"
        )
        
        base_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
            convert_system_message_to_human=True
        )
        
        query_augmentation_chain = condense_prompt | base_llm | StrOutputParser()
        
        if history_text.strip():
            augmented_query = query_augmentation_chain.invoke({
                "history_text": history_text,
                "query": request.query
            })
        else:
            augmented_query = request.query

    except Exception as aug_err:
        raise HTTPException(status_code=500, detail=f"질의 확장 실행 실패: {str(aug_err)}")

    try:
        # 3. 다중 질문 검색기(MultiQueryRetriever) 구성 및 실행
        # 원본/확장된 질의를 바탕으로 LLM이 여러 개의 유사 질문을 추가로 만들어 Chroma DB를 다각도로 검색함
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
            llm=base_llm
        )
        
        retrieved_docs = multi_retriever.invoke(augmented_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "검색된 관련 문서가 존재하지 않습니다."

    except Exception as db_search_err:
        raise HTTPException(status_code=500, detail=f"Chroma DB 다중 질문 검색 실패: {str(db_search_err)}")

    try:
        # 4. 시스템 역할 정의가 포함된 프롬프트 구성 (시험 요구사항 반영)
        rag_prompt = ChatPromptTemplate.from_template(
            "당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요. 문서에 없는 내용은 절대 지어내지 마세요.\n\n"
            "[문서 정보]\n{context}\n\n"
            "[사용자 질문]\n{question}"
        )
        
        rag_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            streaming=True,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        
        rag_chain = rag_prompt | rag_llm | StrOutputParser()
        
        async def generate_stream():
            try:
                async for chunk in rag_chain.astream({"context": context_text, "question": augmented_query}):
                    yield chunk
            except Exception as stream_err:
                yield f"\n\n🍎[최신 코드 배포 성공!]🍎 에러 발생: {str(stream_err)}"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as rag_err:
        raise HTTPException(status_code=500, detail=f"최종 응답 파이프라인 구성 실패: {str(rag_err)}")


@app.get("/")
def read_root():
    return {"message": "🍎[테스트] Chroma DB 및 MultiQueryRetriever 기반 챗봇 서버가 정상 작동 중입니다!!!🍎"}

# Cloud Run 배포를 위한 uvicorn 실행 블록
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
