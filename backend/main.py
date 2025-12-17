import os
import shutil
import uuid
from typing import Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.rag_service import RAGService

# Load environment variables
load_dotenv()

app = FastAPI(title="Lecture RAG Bot API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, specify frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage
# Map session_id -> RAGService instance
sessions: Dict[str, RAGService] = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    message: str

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Create a new session ID
    session_id = str(uuid.uuid4())
    
    # Create a temporary directory for this session if needed, or just use tempfile
    # For simplicity, we'll save to a temp file
    try:
        # Create a temp file
        with open(f"temp_{session_id}.pdf", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        temp_file_path = f"temp_{session_id}.pdf"
        
        # Initialize RAG Service for this session
        rag_service = RAGService()
        success = rag_service.process_pdf(temp_file_path)
        
        if not success:
             raise HTTPException(status_code=500, detail="Failed to process PDF.")
        
        # Store session
        sessions[session_id] = rag_service
        
        # Cleanup temp file
        os.remove(temp_file_path)
        
        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            message="File uploaded and processed successfully."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")
    
    rag_service = sessions[session_id]
    
    try:
        answer = rag_service.get_answer(request.message)
        return ChatResponse(response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
