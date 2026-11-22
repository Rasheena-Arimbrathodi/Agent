from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    content: str
    source_path: str = ""


class RetrievalResult(BaseModel):
    doc_id: str
    title: str
    score: float
    snippet: str


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    output: Any


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question for the agent")
    top_k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    tool_calls: List[ToolCallRecord]
    mode: str
