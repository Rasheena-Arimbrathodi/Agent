from pathlib import Path
from evidenceflow.models import Document
from evidenceflow.retriever import HybridRetriever, tokenize


def test_tokenize_removes_stopwords():
    tokens = tokenize("What is the remote work policy?")
    assert "what" not in tokens
    assert "remote" in tokens
    assert "policy" in tokens


def test_retriever_from_directory():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    retriever = HybridRetriever.from_directory(data_dir)
    assert len(retriever.doc_list) >= 4


def test_retriever_ranking():
    docs = [
        Document(doc_id="d1", title="Stipend", content="Annual stipend is $600 for remote staff."),
        Document(doc_id="d2", title="Security", content="Rotate keys every 90 days."),
    ]
    retriever = HybridRetriever(docs)
    results = retriever.search("remote stipend")
    assert len(results) == 1
    assert results[0].doc_id == "d1"


def test_retriever_empty_query():
    docs = [Document(doc_id="d1", title="Test", content="Sample content")]
    retriever = HybridRetriever(docs)
    assert retriever.search("") == []
    assert retriever.search("the and or") == []


def test_retriever_no_documents():
    retriever = HybridRetriever([])
    assert retriever.search("anything") == []
