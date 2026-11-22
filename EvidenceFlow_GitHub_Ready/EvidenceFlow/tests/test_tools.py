import pytest
from evidenceflow.models import Document
from evidenceflow.retriever import HybridRetriever
from evidenceflow.tools import ToolRegistry, safe_calculate


def test_safe_calculate_basic():
    assert safe_calculate("2 + 3 * 4") == 14.0
    assert safe_calculate("(10 + 5) / 3") == 5.0
    assert safe_calculate("-5 + 12") == 7.0


def test_safe_calculate_division_by_zero():
    with pytest.raises(ValueError):
        safe_calculate("10 / 0")


def test_safe_calculate_blocks_unsafe_code():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('echo hi')")


def test_tool_registry_execution():
    docs = [Document(doc_id="sla", title="SLA", content="Enterprise uptime is 99.95%.")]
    registry = ToolRegistry(HybridRetriever(docs))
    assert set(registry.list_tools()) == {"search_docs", "read_doc", "calculate"}
    search_out = registry.execute("search_docs", query="uptime")
    assert search_out[0]["doc_id"] == "sla"
    doc_out = registry.execute("read_doc", doc_id="sla")
    assert "99.95%" in doc_out["content"]
    missing_out = registry.execute("read_doc", doc_id="unknown")
    assert "error" in missing_out


def test_tool_registry_unknown_tool():
    registry = ToolRegistry(HybridRetriever([]))
    with pytest.raises(KeyError):
        registry.execute("non_existent")
