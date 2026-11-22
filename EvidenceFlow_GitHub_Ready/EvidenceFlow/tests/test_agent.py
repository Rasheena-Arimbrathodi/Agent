from pathlib import Path
from evidenceflow.agent import EvidenceAgent
from evidenceflow.evaluator import run_evaluation
from evidenceflow.retriever import HybridRetriever


def _make_agent() -> EvidenceAgent:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    return EvidenceAgent(HybridRetriever.from_directory(data_dir), use_llm=False)


def test_agent_grounded_answer_with_citation():
    agent = _make_agent()
    res = agent.ask("What is the home-office stipend?")
    assert "policy_remote_work" in res.citations
    assert "$600" in res.answer
    assert any(tc.tool_name == "search_docs" for tc in res.tool_calls)
    assert any(tc.tool_name == "read_doc" for tc in res.tool_calls)


def test_agent_math_tool_invocation():
    agent = _make_agent()
    res = agent.ask("Calculate 25 * 4")
    assert "100" in res.answer
    assert any(tc.tool_name == "calculate" for tc in res.tool_calls)


def test_agent_unanswerable_query():
    agent = _make_agent()
    res = agent.ask("quantum teleportation mars colony")
    assert res.citations == []
    assert "No supporting evidence" in res.answer


def test_evaluation_suite_passes_all_5_cases():
    agent = _make_agent()
    summary = run_evaluation(agent)
    assert summary["total"] == 5
    assert summary["passed"] == 5
    assert summary["pass_rate"] == 1.0
