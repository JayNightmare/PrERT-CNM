from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
from typing import Dict, Any
from pathlib import Path

from models.PrERT.pipeline import PrERTPipeline

app = FastAPI(title="PrERT-CNM Interactive Showcase API", version="1.1")

# Allow the frontend (HTML) to talk to this local server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Global Pipeline State ---
import traceback

pipeline = None
try:
    print("Initializing the PrERT-CNM Core Pipeline (DeBERTa-v3 + ChromaDB)")
    pipeline = PrERTPipeline()
except Exception as e:
    print(f"Error initializing backend: {e}")
    traceback.print_exc()

# --- Task Storage ---
# Stores results as { "task_id": { "status": "processing" | "completed" | "error", "result": {...}, "error": "..." } }
tasks_db: Dict[str, Dict[str, Any]] = {}

class PolicyAnalysisRequest(BaseModel):
    text: str

class TaskResponse(BaseModel):
    task_id: str
    status: str

class TokenHighlight(BaseModel):
    token: str
    weight: float
    category: str

class AuditTrailEntry(BaseModel):
    control_name: str
    triggered_status: bool
    triggered_text: str
    cnm_reason: str
    cause_heatmap: list[TokenHighlight]
    attempts: int

class ViolatedControl(BaseModel):
    control: str
    confidence: float

class AnalysisResult(BaseModel):
    status: str
    total_flags: int
    violated_controls: list[ViolatedControl]
    audit_trail: list[AuditTrailEntry]

class AnalysisStatusResponse(BaseModel):
    task_id: str
    status: str
    message: str | None = None
    result: AnalysisResult | None = None
    error: str | None = None

def run_pipeline_task(task_id: str, content: str | bytes, is_pdf: bool):
    try:
        if pipeline is None:
            raise Exception("PrERT Pipeline failed to initialize locally.")
            
        def status_updater(msg: str):
            if task_id in tasks_db:
                tasks_db[task_id]["message"] = msg
                
        result = pipeline.process_document(content, is_pdf=is_pdf, status_updater=status_updater)
        tasks_db[task_id] = {
            "status": "completed",
            "result": result
        }
    except Exception as e:
        traceback.print_exc()
        tasks_db[task_id] = {
            "status": "error",
            "error": str(e)
        }

@app.post("/analyze", response_model=TaskResponse)
async def analyze_policy_start(request: PolicyAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Starts the analysis in the background and returns a task ID to avoid Cloudflare 524 timeouts.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="PrERT Pipeline failed to initialize locally.")
        
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {"status": "processing"}
    background_tasks.add_task(run_pipeline_task, task_id, request.text, False)
    
    return TaskResponse(task_id=task_id, status="processing")

@app.post("/analyze/file", response_model=TaskResponse)
async def analyze_file_start(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Starts file analysis in the background and returns a task ID.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="PrERT Pipeline failed to initialize locally.")
        
    try:
        content = await file.read()
        is_pdf = file.filename.lower().endswith(".pdf")
        
        if not is_pdf:
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Text files must be UTF-8 encoded.")
                
        task_id = str(uuid.uuid4())
        tasks_db[task_id] = {"status": "processing"}
        background_tasks.add_task(run_pipeline_task, task_id, content, is_pdf)
        
        return TaskResponse(task_id=task_id, status="processing")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to start file analysis: {str(e)}")

@app.get("/analyze/status/{task_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(task_id: str):
    """
    Poll this endpoint to get the status of the background task.
    """
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task_info = tasks_db[task_id]
    
    return AnalysisStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        message=task_info.get("message"),
        result=task_info.get("result"),
        error=task_info.get("error")
    )

@app.get("/health")
async def health_check():
    import uvicorn
    return {"status": "ok", "pipeline_initialized": pipeline is not None, "server_version": os.getenv("SERVER_VERSION", "unknown"), "uvicorn_version": uvicorn.__version__}
