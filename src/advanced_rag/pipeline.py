"""Build the whole knowledge base (parse -> chunk -> index -> graph)."""
from pathlib import Path

from . import config
from .chunking import make_chunks
from .graph import build_graph
from .indexing import build_index
from .parsing import load_docs
from .retrieval import Retriever


def build_retriever(pdf_dir: Path = config.PDF_DIR) -> tuple[Retriever, dict, object]:
    docs = load_docs(Path(pdf_dir))
    parents, children = make_chunks(docs)
    print("parents:", len(parents), "children:", len(children))
    index = build_index(children)
    graph = build_graph(docs)
    return Retriever(children, parents, index, graph), docs, graph
