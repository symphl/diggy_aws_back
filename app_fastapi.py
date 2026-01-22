# app_fastapi.py
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import your existing processing functions
# Make sure these functions are compatible with being called from this script.
# For example, they should not rely on Streamlit's session state.
from processing import (
    run_full_pipeline,
    summarize_document,
    process_image_for_description,
    summarize_video,
    summarize_url,
)
from api_clients import answer_followup

# --- Pydantic Models for Request/Response Bodies ---

class ProcessRequest(BaseModel):
    query: str
    context: Optional[str] = None

class SummarizeUrlRequest(BaseModel):
    url: str

class FollowupRequest(BaseModel):
    question: str
    context: str

class ErrorResponse(BaseModel):
    detail: str

# --- FastAPI Application ---

app = FastAPI(
    title="Diggi News Assistant API",
    description="API for processing news topics, URLs, documents, and more.",
    version="1.0.0",
)

# --- API Endpoints ---

@app.post(
    "/process",
    summary="Process a general query",
    response_description="Full analysis including articles, summary, and perspectives.",
)
async def process_query(request: ProcessRequest):
    """
    The main entry point for analysis. It takes a text query and returns a full breakdown.
    """
    try:
        articles, summary, followups, perspectives, error = run_full_pipeline(
            request.query, context=request.context
        )
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        return {
            "articles": articles,
            "summary": summary,
            "followups": followups,
            "perspectives": perspectives,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/summarize/url",
    summary="Summarize a URL",
    response_description="A summary of the content at the given URL.",
)
async def summarize_url_endpoint(request: SummarizeUrlRequest):
    """
    Takes a URL and returns a summary of the article.
    """
    try:
        # Assuming summarize_url returns the summary string directly
        summary = summarize_url(request.url)
        if not summary:
            raise HTTPException(status_code=404, detail="Could not generate summary for the URL.")
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/summarize/document",
    summary="Summarize a Document",
    response_description="A summary of the uploaded document.",
)
async def summarize_document_endpoint(doc_file: UploadFile = File(...)):
    """
    Accepts a document upload (.pdf, .docx) and returns a summary.
    """
    if doc_file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Invalid document type. Please upload a PDF or DOCX file.")
    try:
        # The summarize_document function from your `processing.py` might expect a file path
        # or a file-like object. We will save the uploaded file temporarily.
        summary = summarize_document(doc_file.file)
        if not summary:
            raise HTTPException(status_code=500, detail="Failed to generate summary from document.")
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/analyze/image",
    summary="Analyze an Image",
    response_description="A description of the uploaded image.",
)
async def analyze_image_endpoint(img_file: UploadFile = File(...)):
    """
    Accepts an image upload (.png, .jpg, .jpeg) and returns a description.
    """
    if img_file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Please upload a PNG or JPG file.")
    try:
        # Similar to the document, we pass the file-like object.
        description = process_image_for_description(img_file.file)
        if not description:
            raise HTTPException(status_code=500, detail="Failed to generate description from image.")
        return {"description": description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/summarize/video",
    summary="Summarize a Video",
    response_description="A summary of the uploaded video.",
)
async def summarize_video_endpoint(vid_file: UploadFile = File(...)):
    """
    Accepts a video upload (.mp4, .mov, .avi) and returns a summary.
    """
    if vid_file.content_type not in ["video/mp4", "video/quicktime", "video/x-msvideo"]:
        raise HTTPException(status_code=400, detail="Invalid video type. Please upload an MP4, MOV, or AVI file.")
    try:
        # summarize_video might need a file path. Let's save it temporarily.
        summary, _ = summarize_video(vid_file.file) # Assuming it returns a tuple (summary, some_other_data)
        if not summary:
            raise HTTPException(status_code=500, detail="Failed to generate summary from video.")
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/followup",
    summary="Answer a Follow-up Question",
    response_description="An answer to the follow-up question based on the provided context.",
)
async def followup_endpoint(request: FollowupRequest):
    """
    Takes a question and a context summary to provide an answer.
    """
    try:
        answer = answer_followup(request.question, context=request.context)
        if not answer:
            raise HTTPException(status_code=404, detail="Could not generate an answer.")
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run this API, use the command:
# uvicorn app_fastapi:app --reload
