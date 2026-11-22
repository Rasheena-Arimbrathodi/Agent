from pathlib import Path
from fastapi import FastAPI
from evidenceflow.agent import EvidenceAgent
from evidenceflow.config import settings
from evidenceflow.evaluator import run_evaluation
from evidenceflow.models import AskRequest, AskResponse
from evidenceflow.retriever import HybridRetriever

app = FastAPI(
    title="EvidenceFlow — Agentic AI Assistant",
    description="Evidence-grounded AI agent with BM25 + TF-IDF retrieval, safe tool calling, and evaluation.",
    version="0.1.0",
)

_data_dir = Path(settings.default_data_dir)
_retriever = HybridRetriever.from_directory(_data_dir)
_agent = EvidenceAgent(_retriever)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "documents_indexed": len(_retriever.doc_list),
        "mode": "llm" if _agent.use_llm else "offline-deterministic",
    }


@app.get("/documents")
def list_documents():
    return [
        {"doc_id": d.doc_id, "title": d.title, "source_path": d.source_path}
        for d in _retriever.doc_list
    ]


@app.post("/ask", response_model=AskResponse)
def ask_agent(payload: AskRequest):
    return _agent.ask(query=payload.query, top_k=payload.top_k)


@app.get("/evaluate")
def evaluate_agent():
    return run_evaluation(_agent)
