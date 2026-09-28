import logging
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import agentic_mode_enabled, analyze_draft

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Research Citation Agent")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.middleware("http")
async def revalidate_frontend(request: Request, call_next):
    # Without this, browsers heuristically cache app.js and keep showing an
    # old UI after a deploy. "no-cache" still allows ETag revalidation.
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


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
    results, truncated = analyze_draft(req.text)
    return {"results": results, "truncated": truncated, "agentic_mode": agentic_mode_enabled()}
