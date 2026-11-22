import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List
from evidenceflow.models import Document, RetrievalResult

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "this",
    "that", "it", "as", "from", "what", "how", "when", "where", "who", "which"
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class HybridRetriever:
    """Combines BM25 and TF-IDF cosine similarity for local document retrieval."""

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        self.documents: Dict[str, Document] = {d.doc_id: d for d in documents}
        self.doc_list: List[Document] = list(documents)
        self.k1 = k1
        self.b = b
        self._build_index()

    @classmethod
    def from_directory(cls, directory: str | Path) -> "HybridRetriever":
        path = Path(directory)
        docs: List[Document] = []
        if path.exists():
            for file_path in sorted(path.glob("*.md")):
                text = file_path.read_text(encoding="utf-8")
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                title = lines[0].lstrip("# ").strip() if lines else file_path.stem
                docs.append(
                    Document(
                        doc_id=file_path.stem,
                        title=title,
                        content=text,
                        source_path=str(file_path),
                    )
                )
        return cls(docs)

    def _build_index(self) -> None:
        self.doc_tokens = [tokenize(f"{d.title} {d.content}") for d in self.doc_list]
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        self.n_docs = len(self.doc_list)

        df: Counter = Counter()
        for freqs in self.doc_freqs:
            for term in freqs:
                df[term] += 1
        self.df = df

        # Precompute TF-IDF vectors
        self.tfidf_vecs: List[Dict[str, float]] = []
        self.tfidf_norms: List[float] = []
        for freqs in self.doc_freqs:
            vec: Dict[str, float] = {}
            for term, tf in freqs.items():
                idf = math.log((self.n_docs + 1) / (self.df[term] + 1)) + 1.0
                vec[term] = (1 + math.log(tf)) * idf
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.tfidf_vecs.append(vec)
            self.tfidf_norms.append(norm)

    def _bm25_score(self, query_tokens: List[str], idx: int) -> float:
        score = 0.0
        freqs = self.doc_freqs[idx]
        doc_len = self.doc_lens[idx]
        for term in query_tokens:
            if term not in freqs:
                continue
            tf = freqs[term]
            df = self.df[term]
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
            score += idf * ((tf * (self.k1 + 1)) / (denom or 1.0))
        return score

    def _tfidf_cosine(self, query_tokens: List[str], idx: int) -> float:
        q_counts = Counter(query_tokens)
        q_vec: Dict[str, float] = {}
        for term, tf in q_counts.items():
            if term in self.df:
                idf = math.log((self.n_docs + 1) / (self.df[term] + 1)) + 1.0
                q_vec[term] = (1 + math.log(tf)) * idf
        if not q_vec:
            return 0.0
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        doc_vec = self.tfidf_vecs[idx]
        dot = sum(q_val * doc_vec.get(term, 0.0) for term, q_val in q_vec.items())
        return dot / (q_norm * self.tfidf_norms[idx])

    def search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        if not self.doc_list:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        bm25_raw = [self._bm25_score(q_tokens, i) for i in range(self.n_docs)]
        max_bm25 = max(bm25_raw) or 1.0
        tfidf_raw = [self._tfidf_cosine(q_tokens, i) for i in range(self.n_docs)]

        scored: List[RetrievalResult] = []
        for i, doc in enumerate(self.doc_list):
            norm_bm25 = bm25_raw[i] / max_bm25 if max(bm25_raw) > 0 else 0.0
            hybrid_score = round(0.6 * norm_bm25 + 0.4 * tfidf_raw[i], 4)
            if hybrid_score > 0:
                snippet = self._extract_snippet(doc.content, q_tokens)
                scored.append(
                    RetrievalResult(
                        doc_id=doc.doc_id,
                        title=doc.title,
                        score=hybrid_score,
                        snippet=snippet,
                    )
                )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def _extract_snippet(self, content: str, q_tokens: List[str]) -> str:
        lines = [ln.strip("- ").strip() for ln in content.splitlines() if ln.strip() and not ln.startswith("#")]
        best_line = lines[0] if lines else content[:180]
        best_hits = -1
        for line in lines:
            l_toks = set(tokenize(line))
            hits = len(l_toks.intersection(q_tokens))
            if hits > best_hits:
                best_hits = hits
                best_line = line
        return best_line
