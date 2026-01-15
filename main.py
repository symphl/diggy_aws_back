from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
from processing import run_full_pipeline, extract_article
from api_clients import summarize_text, extract_keywords, transcribe_audio_groq, extract_text_from_pdf, extract_text_from_docx, transcribe_audio_local
import sys


# Allow main.py to see modules in the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from processing import run_full_pipeline
# ... rest of your code


app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None

@app.get("/")
def home():
    return {"message": "Diggi API is running"}

@app.post("/analyze")
async def analyze_text(data: QueryRequest):
    """Endpoint for simple text query or URL"""
    query = data.query
    # If it's a URL, extract keywords first
    if query.startswith("http"):
        text = extract_article(query)
        summary = summarize_text(text)
        query = extract_keywords(summary)
    
    result = run_full_pipeline(query, context=data.context)
    return result

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...), context: Optional[str] = Form(None)):
    """Endpoint for Video/Image/Document uploads"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Determine how to get keywords based on file type
        filename = file.filename.lower()
        print(f"Processing file: {filename}, Content-Type: {file.content_type}")

        if file.content_type.startswith("video") or file.content_type.startswith("audio"):
            # Use local Whisper for audio/video
            transcription = transcribe_audio_local(tmp_path)
            summary = summarize_text(transcription)
            keywords = extract_keywords(summary)
        
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(tmp_path)
            if not text:
                raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
            summary = summarize_text(text)
            keywords = extract_keywords(summary)

        elif filename.endswith(".docx") or filename.endswith(".doc"):
            text = extract_text_from_docx(tmp_path)
            if not text:
                raise HTTPException(status_code=400, detail="Could not extract text from DOCX.")
            summary = summarize_text(text)
            keywords = extract_keywords(summary)
            
        else:
            # Fallback or other image handling if previously present, but for now just text
             # Add logic for PDFs/Images here similar to your processing.py
            keywords = "latest news" 

        result = run_full_pipeline(keywords, context=context)
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)