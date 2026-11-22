import re
from typing import List, Optional
from evidenceflow.config import settings
from evidenceflow.models import AskResponse, ToolCallRecord
from evidenceflow.retriever import HybridRetriever
from evidenceflow.tools import ToolRegistry

MATH_PATTERN = re.compile(r"(?:calculate|compute|what is)\s+([0-9\.\+\-\*\/\(\)\s]+)", re.IGNORECASE)
PURE_EXPR_PATTERN = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")


class EvidenceAgent:
    """Evidence-grounded AI agent with deterministic offline orchestration."""

    def __init__(self, retriever: HybridRetriever, use_llm: Optional[bool] = None):
        self.retriever = retriever
        self.tools = ToolRegistry(retriever)
        self.use_llm = settings.llm_enabled if use_llm is None else use_llm

    def _extract_math_expression(self, query: str) -> Optional[str]:
        stripped = query.strip().rstrip("?")
        if PURE_EXPR_PATTERN.match(stripped) and any(op in stripped for op in "+-*/"):
            return stripped
        match = MATH_PATTERN.search(stripped)
        if match:
            candidate = match.group(1).strip()
            if any(op in candidate for op in "+-*/") and any(ch.isdigit() for ch in candidate):
                return candidate
        return None

    def ask(self, query: str, top_k: int = 3) -> AskResponse:
        tool_records: List[ToolCallRecord] = []
        citations: List[str] = []
        mode = "llm" if self.use_llm else "offline-deterministic"

        # Step 1: Check for safe arithmetic tool usage
        expr = self._extract_math_expression(query)
        calc_note = ""
        if expr:
            try:
                calc_out = self.tools.execute("calculate", expression=expr)
                tool_records.append(
                    ToolCallRecord(
                        tool_name="calculate",
                        arguments={"expression": expr},
                        output=calc_out,
                    )
                )
                calc_note = f"Calculated result for `{expr}` is **{calc_out['result']}**."
            except ValueError:
                pass

        # Step 2: Search knowledge base
        search_results = self.tools.execute("search_docs", query=query, top_k=top_k)
        tool_records.append(
            ToolCallRecord(
                tool_name="search_docs",
                arguments={"query": query, "top_k": top_k},
                output=search_results,
            )
        )

        # Step 3: Read top matching document for full context grounding
        evidence_lines: List[str] = []
        if search_results:
            top_doc_id = search_results[0]["doc_id"]
            full_doc = self.tools.execute("read_doc", doc_id=top_doc_id)
            tool_records.append(
                ToolCallRecord(
                    tool_name="read_doc",
                    arguments={"doc_id": top_doc_id},
                    output={"doc_id": top_doc_id, "title": full_doc.get("title", "")},
                )
            )
            for res in search_results:
                doc_id = res["doc_id"]
                if doc_id not in citations:
                    citations.append(doc_id)
                evidence_lines.append(f"- {res['snippet']} [source: {doc_id}]")

        # Step 4: Synthesize grounded response
        if calc_note and not evidence_lines:
            answer = calc_note
        elif evidence_lines:
            prefix = f"{calc_note}

" if calc_note else ""
            answer = (
                f"{prefix}Based on the retrieved knowledge base evidence:
"
                + "
".join(evidence_lines)
            )
        else:
            answer = (
                "No supporting evidence was found in the local knowledge base for that query."
            )

        return AskResponse(
            query=query,
            answer=answer,
            citations=citations,
            tool_calls=tool_records,
            mode=mode,
        )
