"""FastAPI: thin wrappers invoking LangGraph graphs."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (for `src.*`)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import shutil, os
from src.schemas.models import QuizQuestion

app = FastAPI(title="Learning Companion (LangGraph RAG)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

DATA = "data/uploads"
os.makedirs(DATA, exist_ok=True)


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    student_id: str = "anon"
    topic: str | None = Field(default=None, max_length=200)


class QuizIn(BaseModel):
    scope: str = Field(min_length=1, max_length=200)
    student_id: str = "anon"
    difficulty: str = "auto"
    n: int = Field(default=5, ge=1, le=8)


class QuizSubmitIn(BaseModel):
    scope: str = Field(min_length=1, max_length=200)
    student_id: str = "anon"
    questions: list[QuizQuestion] = Field(min_length=1, max_length=8)
    answers: dict[str, str]


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    filename = Path(file.filename or "upload").name
    supported = {".pdf", ".ppt", ".pptx", ".txt", ".srt", ".vtt", ".jpg", ".jpeg", ".png"}
    if Path(filename).suffix.lower() not in supported:
        raise HTTPException(status_code=415, detail="Supported files: PDF, PPT, PPTX, TXT, SRT, VTT, JPG, JPEG, PNG.")

    from src.graphs.ingest_graph import ingest_file
    dest = os.path.join(DATA, filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    out = ingest_file(dest)
    if out["errors"]:
        raise HTTPException(status_code=422, detail="; ".join(out["errors"]))
    if not out["units"]:
        raise HTTPException(status_code=422, detail="No readable content was found in the uploaded file.")
    return {"file": filename, "kind": out["kind"],
            "units": len(out["units"]), "errors": []}


@app.get("/health")
def health():
    from src.config.settings import get_settings
    from src.stores.graph import get_graph

    settings = get_settings()
    vector_backend = settings.VECTOR_BACKEND.lower()
    if vector_backend == "chroma":
        storage_state = "directory present · contents unverified" if Path(settings.CHROMA_DIR).exists() else "awaiting first document"
    else:
        storage_state = "URL configured" if settings.QDRANT_URL else "not configured"
    return {
        "graph_backend": "NetworkX",
        "graph_nodes": get_graph().number_of_nodes(),
        "vector_backend": vector_backend,
        "vector_storage_state": storage_state,
        "retrieval_mode": "Hybrid · semantic + BM25 + graph context",
    }


@app.post("/chat")
def chat(inp: ChatIn):
    from src.graphs.tutor_graph import ask
    r = ask(inp.question, student_id=inp.student_id, topic=inp.topic)
    return {"answer": r["answer"], "verdict": r["verdict"], "citations": r["citations"]}


@app.post("/quiz")
def quiz(inp: QuizIn):
    from src.graphs.quiz_graph import build_quiz_graph
    r = build_quiz_graph().invoke({"student_id": inp.student_id, "scope": inp.scope,
                                   "difficulty_target": inp.difficulty, "n_questions": inp.n,
                                   "questions": [], "results": [], "weak_topics": []})
    return {"questions": r["questions"]}


@app.post("/quiz/submit")
def submit_quiz(inp: QuizSubmitIn):
    from src.graphs.quiz_graph import n_feedback

    state = {
        "student_id": inp.student_id,
        "scope": inp.scope,
        "difficulty_target": "auto",
        "n_questions": len(inp.questions),
        "questions": [question.model_dump() for question in inp.questions],
        "results": [],
        "weak_topics": [],
    }
    result = n_feedback(state, answers=inp.answers)
    return {"results": result["results"], "weak_topics": result["weak_topics"]}


@app.get("/mastery/{student_id}")
def mastery(student_id: str):
    from src.stores.learner import all_mastery
    return all_mastery(student_id)
