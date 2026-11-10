"""CLI entry point.

Examples
--------
Ask one question:
    python scripts/run_pipeline.py --query "What is the liability cap?"

Run the full evaluation (retrieval check + RAGAS):
    python scripts/run_pipeline.py --eval

Also draw the knowledge graph / run DSPy:
    python scripts/run_pipeline.py --eval --graph --dspy
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advanced_rag import config  # noqa: E402
from advanced_rag.evaluation import (generate_answers, llm_judge, load_eval_set,  # noqa: E402
                                     retrieval_check, run_dspy, run_ragas)
from advanced_rag.graph import draw_graph  # noqa: E402
from advanced_rag.pipeline import build_retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Advanced RAG for contracts")
    ap.add_argument("--pdf-dir", default=str(config.PDF_DIR), help="folder with contract PDFs")
    ap.add_argument("--query", help="ask a single question")
    ap.add_argument("--eval", action="store_true", help="run retrieval + answer evaluation")
    ap.add_argument("--graph", action="store_true", help="save knowledge graph PNG to outputs/")
    ap.add_argument("--dspy", action="store_true", help="run DSPy few-shot optimisation")
    args = ap.parse_args()

    retriever, _, graph = build_retriever(Path(args.pdf_dir))

    if args.graph:
        draw_graph(graph)
        print("saved outputs/knowledge_graph.png")

    if args.query:
        res = retriever.ask(args.query)
        print("\nTop sources:", res["sources"])
        print("\n" + res["answer"])

    if args.eval:
        eval_set = load_eval_set()
        retrieval_check(retriever, eval_set)
        if generate_answers(retriever, eval_set):
            if run_ragas(eval_set) is None:
                llm_judge(eval_set)
            if args.dspy:
                run_dspy(eval_set)
        else:
            print("LLM unavailable: check API keys in .env")

    if not (args.query or args.eval or args.graph):
        ap.print_help()


if __name__ == "__main__":
    main()
