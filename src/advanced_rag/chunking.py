"""Parent/child chunking: small children for search, big parents for context."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def make_chunks(docs: dict[str, str],
                parent_size: int = config.PARENT_CHUNK_SIZE,
                child_size: int = config.CHILD_CHUNK_SIZE):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=50)

    parents: dict[str, str] = {}
    children: list[dict] = []
    for name, text in docs.items():
        for i, ptext in enumerate(parent_splitter.split_text(text)):
            pid = f"{name}::p{i}"
            parents[pid] = ptext
            for ctext in child_splitter.split_text(ptext):
                children.append({"text": ctext, "parent_id": pid, "source": name})
    return parents, children
