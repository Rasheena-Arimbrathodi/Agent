from pathlib import Path
from evidenceflow.agent import EvidenceAgent
from evidenceflow.retriever import HybridRetriever


def main():
    data_path = Path(__file__).resolve().parents[1] / "data"
    retriever = HybridRetriever.from_directory(data_path)
    agent = EvidenceAgent(retriever, use_llm=False)

    queries = [
        "What is the home-office stipend?",
        "What is the Enterprise uptime SLA?",
        "Calculate 125 * 8",
    ]
    for q in queries:
        res = agent.ask(q)
        print(f"Q: {q}")
        print(f"A: {res.answer}")
        print(f"Citations: {res.citations}
")


if __name__ == "__main__":
    main()
