from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path

from prert.pipeline import PrERTPipeline

app = FastAPI(title="PrERT-CNM Interactive Showcase API", version="1.0")

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

class PolicyAnalysisRequest(BaseModel):
    text: str

class TokenHighlight(BaseModel):
    token: str
    weight: float
    category: str  # 'red' (trigger), 'green' (context), 'blue' (safe)

class AuditTrailEntry(BaseModel):
    control_name: str
    triggered_status: bool
    triggered_text: str
    cnm_reason: str
    cause_heatmap: list[TokenHighlight]

class ViolatedControl(BaseModel):
    control: str
    confidence: float

class AnalysisResult(BaseModel):
    status: str
    total_flags: int
    violated_controls: list[ViolatedControl]
    audit_trail: list[AuditTrailEntry]

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_policy(request: PolicyAnalysisRequest):
    """
    Real endpoint invoking the PrERT-CNM inference pipeline.
    It passes the document through Semantic Context Chunking, DeBERTa Encoding,
    Attention Rollout, and finally Custom Multi-Label Classification.
    """
    try:
        if pipeline is None:
            raise HTTPException(status_code=500, detail="PrERT Pipeline failed to initialize locally. Check server logs (e.g. invalid ChromaDB credentials).")
        # Pass document to the stateful pipeline
        result = pipeline.process_document(request.text, is_pdf=False)
        return AnalysisResult(**result)
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline inference error: {str(e)}")

@app.post("/analyze/file", response_model=AnalysisResult)
async def analyze_file(file: UploadFile = File(...)):
    """
    Endpoint for uploading raw documents (.txt or .pdf).
    """
    try:
        if pipeline is None:
            raise HTTPException(status_code=500, detail="PrERT Pipeline failed to initialize locally.")
            
        content = await file.read()
        is_pdf = file.filename.lower().endswith(".pdf")
        
        if not is_pdf:
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Text files must be UTF-8 encoded.")
                
        result = pipeline.process_document(content, is_pdf=is_pdf)
        return AnalysisResult(**result)
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline inference error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "pipeline": "initialized" if 'pipeline' in globals() else "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
