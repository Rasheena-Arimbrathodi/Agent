from dataclasses import dataclass
from typing import Any, Dict, List
from evidenceflow.agent import EvidenceAgent


@dataclass
class EvalCase:
    case_id: str
    query: str
    expected_doc_id: str
    expected_substring: str


DEFAULT_EVAL_CASES: List[EvalCase] = [
    EvalCase(
        case_id="eval_01_remote_stipend",
        query="What is the home-office stipend for remote work?",
        expected_doc_id="policy_remote_work",
        expected_substring="$600",
    ),
    EvalCase(
        case_id="eval_02_enterprise_sla",
        query="What is the Enterprise tier monthly uptime guarantee?",
        expected_doc_id="product_sla",
        expected_substring="99.95%",
    ),
    EvalCase(
        case_id="eval_03_key_rotation",
        query="How often must production API keys be rotated?",
        expected_doc_id="security_guidelines",
        expected_substring="90 days",
    ),
    EvalCase(
        case_id="eval_04_latency_release",
        query="What was the median API latency improvement in v2.4?",
        expected_doc_id="release_notes_v2",
        expected_substring="85ms",
    ),
    EvalCase(
        case_id="eval_05_math_tool",
        query="Calculate 600 * 3",
        expected_doc_id="",
        expected_substring="1800",
    ),
]


def run_evaluation(
    agent: EvidenceAgent, cases: List[EvalCase] | None = None
) -> Dict[str, Any]:
    eval_cases = cases or DEFAULT_EVAL_CASES
    results = []
    passed = 0

    for case in eval_cases:
        resp = agent.ask(case.query)
        citation_ok = (
            True
            if not case.expected_doc_id
            else case.expected_doc_id in resp.citations
        )
        content_ok = case.expected_substring.lower() in resp.answer.lower()
        ok = citation_ok and content_ok
        if ok:
            passed += 1
        results.append(
            {
                "case_id": case.case_id,
                "passed": ok,
                "citation_ok": citation_ok,
                "content_ok": content_ok,
                "citations": resp.citations,
            }
        )

    total = len(eval_cases)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "cases": results,
    }
