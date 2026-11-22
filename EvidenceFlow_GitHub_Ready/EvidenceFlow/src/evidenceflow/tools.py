import ast
import operator
from typing import Any, Callable, Dict, List
from evidenceflow.retriever import HybridRetriever

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression: str) -> float:
    """Safely evaluate basic arithmetic expressions using AST parsing."""
    cleaned = expression.strip()
    if not cleaned:
        raise ValueError("Empty expression")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("Division by zero")
            return float(_ALLOWED_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return float(_ALLOWED_OPS[type(node.op)](_eval(node.operand)))
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    tree = ast.parse(cleaned, mode="eval")
    result = _eval(tree)
    return round(result, 6)


class ToolRegistry:
    """Registry of agent tools for search, reading documents, and safe math."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self._tools: Dict[str, Callable[..., Any]] = {
            "search_docs": self.search_docs,
            "read_doc": self.read_doc,
            "calculate": self.calculate,
        }

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def execute(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self._tools:
            raise KeyError(f"Unknown tool: {tool_name}")
        return self._tools[tool_name](**kwargs)

    def search_docs(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results = self.retriever.search(query=query, top_k=top_k)
        return [r.model_dump() for r in results]

    def read_doc(self, doc_id: str) -> Dict[str, Any]:
        doc = self.retriever.documents.get(doc_id)
        if not doc:
            return {"error": f"Document '{doc_id}' not found"}
        return doc.model_dump()

    def calculate(self, expression: str) -> Dict[str, Any]:
        value = safe_calculate(expression)
        return {"expression": expression, "result": value}
