import os
import io
import uvicorn
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from supabase import create_client, Client

# Langchain 컴포넌트
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 로컬 개발 환경용 변수 로드
load_dotenv()

# 환경 변수에서 구글 AI API 키 등 중요 인증 정보 로드
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# 필수 환경 변수 누락 여부 강도 높은 검증
required_keys = ["GOOGLE_API_KEY", "OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
missing_keys = [key for key in required_keys if not os.environ.get(key)]
if missing_keys:
    raise ValueError(f"서버 초기화 치명적 실패! 정확히 다음 변수가 누락되었습니다: {missing_keys}")

try:
    # 벡터 DB 개발(Supabase) 클라이언트 초기화 및 오픈 AI 임베딩 모델 객체 생성
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
except Exception as e:
    raise RuntimeError(f"클라이언트 객체 초기화 중 시스템 에러 발생. 상세 원인: {str(e)}")

app = FastAPI(title="도시 정책 RAG API 서버")

# 웹 클라이언트(GAS 등)에서의 직접 API 통신을 허용하기 위한 CORS 미들웨어 적용
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
    """입력한 PDF 파일을 텍스트를 변환하고 청크 단위로 쪼개기 및 벡터 DB 적재"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail=f"요청 거부됨: 확장자가 유효하지 않습니다. '.pdf' 파일을 업로드해야 합니다.")
    
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
            raise HTTPException(status_code=422, detail="문서 분석 실패: 업로드된 PDF 파일 내부에서 추출 가능한 텍스트를 찾지 못했습니다.")

        # 텍스트를 청크 단위로 쪼개기 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(full_text)
        
        # --- [수정된 부분: 단건 처리 -> 배치(Batch) 처리로 통신 병목 해결] ---
        try:
            embedding_vectors = embeddings.embed_documents(chunks)
        except Exception as emb_err:
            raise HTTPException(status_code=500, detail=f"임베딩 일괄 변환 실패: {str(emb_err)}")

        bulk_data = []
        for i, (chunk, vector) in enumerate(zip(chunks, embedding_vectors)):
            bulk_data.append({
                "content": chunk,
                "metadata": {"source": file.filename, "chunk_index": i},
                "embedding": vector
            })

        batch_size = 100
        for i in range(0, len(bulk_data), batch_size):
            batch = bulk_data[i : i + batch_size]
            try:
                supabase.table("document_chunks").insert(batch).execute()
            except Exception as db_err:
                raise HTTPException(status_code=500, detail=f"Supabase 데이터 적재 실패 (청크 {i} ~ {i+len(batch)-1}): {str(db_err)}")
                
        return {"status": "success", "message": f"총 {len(chunks)}개의 청크 조각이 안전하게 벡터 DB에 적재되었습니다."}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"알 수 없는 서버 오류: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """질의 확장 처리 후 Supabase 리트리버를 통해 관련 문서를 찾고 스트리밍 답변을 제공"""
    
    # --- [수정된 부분: 대화 기록을 단순 문자열로 변환하여 SystemMessage 에러 원천 차단] ---
    history_text = ""
    for msg in request.chat_history:
        role = "사용자" if msg.get("role") == "user" else "AI"
        history_text += f"{role}: {msg.get('content')}\n"

    try:
        # 1. SystemMessage를 생성하지 않는 단일 텍스트 프롬프트 방식 적용
        condense_prompt = ChatPromptTemplate.from_template(
            "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라.\n\n"
            "[대화 기록]\n{history_text}\n\n"
            "[현재 질문]\n{query}"
        )
        
        condense_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
            convert_system_message_to_human=True  # <-- 이 옵션을 꼭 추가!
        )
        
        query_augmentation_chain = condense_prompt | condense_llm | StrOutputParser()
        
        if history_text.strip():
            augmented_query = query_augmentation_chain.invoke({
                "history_text": history_text,
                "query": request.query
            })
        else:
            augmented_query = request.query

    except Exception as aug_err:
        raise HTTPException(status_code=500, detail=f"질의 확장(Query Augmentation) 실행 실패: {str(aug_err)}")

    try:
        # 2. 증강된 질의로 벡터 DB 검색
        query_embedding = embeddings.embed_query(augmented_query)
        
        rpc_response = supabase.rpc(
            "match_document_chunks",
            {"query_embedding": query_embedding, "match_threshold": 0.7, "match_count": 5}
        ).execute()
        
        retrieved_docs = rpc_response.data
        context_text = "\n\n".join([doc["content"] for doc in retrieved_docs]) if retrieved_docs else "검색된 관련 문서가 존재하지 않습니다."

    except Exception as db_search_err:
        raise HTTPException(status_code=500, detail=f"벡터 리트리버 검색 실패: {str(db_search_err)}")

    try:
        # 3. RAG 프롬프트 역시 SystemMessage가 생기지 않는 단일 텍스트로 변경
        rag_prompt = ChatPromptTemplate.from_template(
            "너는 문서에 기반해 답변하는 도시 정책 전문가야. 다음 제공된 문서 정보를 바탕으로 사용자의 질문에 전문적으로 답변해. 문서에 없는 내용은 지어내지 마.\n\n"
            "[문서 정보]\n{context}\n\n"
            "[사용자 질문]\n{question}"
        )
        
       rag_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            streaming=True,
            temperature=0.2,
            convert_system_message_to_human=True  # <-- 이 옵션을 꼭 추가!
        )
        
        rag_chain = rag_prompt | rag_llm | StrOutputParser()
        
        async def generate_stream():
            try:
                async for chunk in rag_chain.astream({"context": context_text, "question": augmented_query}):
                    yield chunk
            except Exception as stream_err:
                # 에러 메시지 앞에 사과 이모티콘과 테스트 문구를 추가합니다.
                yield f"\n\n🍎[최신 코드 배포 성공!]🍎 에러 발생: {str(stream_err)}"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as rag_err:
        raise HTTPException(status_code=500, detail=f"최종 응답 파이프라인 구성 실패: {str(rag_err)}")

@app.get("/")
def read_root():
    return {"message": "🍎[테스트] 드디어 최신 코드가 서버에 반영되었습니다!!!🍎"}

# Cloud Run 배포를 위한 uvicorn 실행 블록
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
