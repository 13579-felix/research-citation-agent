from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import agentic_mode_enabled, analyze_draft

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Research Citation Agent")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/status")
def status():
    return {"agentic_mode": agentic_mode_enabled()}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    return {"results": analyze_draft(req.text), "agentic_mode": agentic_mode_enabled()}
