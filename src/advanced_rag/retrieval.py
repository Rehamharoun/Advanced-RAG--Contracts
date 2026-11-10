"""Retrieval: multi-query expansion -> hybrid (dense + BM25, RRF) -> graph boost
-> parent-chunk expansion -> grounded answer."""
from collections import defaultdict

import numpy as np

from . import config
from .graph import docs_for_query
from .llm import ask_llm


class Retriever:
    def __init__(self, children, parents, index, graph):
        self.children = children
        self.parents = parents
        self.index = index
        self.graph = graph

    def expand_query(self, query: str) -> list[str]:
        out = [query]
        reply = ask_llm(
            "Rewrite this contract question in 3 different ways, one per line, no numbering:\n" + query
        )
        if reply:
            out += [l.strip("-• ").strip() for l in reply.splitlines() if l.strip()][:3]
        else:  # offline fallback
            kw = " ".join(w for w in config.tokenize(query) if w not in config.STOP_WORDS)
            out += [kw, f"contract clause about {kw}"]
        return out

    def hybrid_retrieve(self, query: str, top_k: int = 8, k: int = 60) -> dict[int, float]:
        """Reciprocal Rank Fusion of dense and BM25 results."""
        q_vec = self.index.embed_model.encode(query).tolist()
        res = self.index.qdrant.query_points(config.COLLECTION, query=q_vec, limit=top_k)
        vec_ids = [p.id for p in res.points]

        scores = self.index.bm25.get_scores(config.tokenize(query))
        bm_ids = [int(i) for i in np.argsort(scores)[::-1][:top_k] if scores[i] > 0]

        fused: dict[int, float] = defaultdict(float)
        for rank, cid in enumerate(vec_ids):
            fused[cid] += 1.0 / (k + rank + 1)
        for rank, cid in enumerate(bm_ids):
            fused[cid] += 1.0 / (k + rank + 1)
        return fused

    def retrieve(self, query: str, top_k: int = 10):
        fused: dict[int, float] = defaultdict(float)
        for q in self.expand_query(query):
            for cid, s in self.hybrid_retrieve(q).items():
                fused[cid] += s

        boost = docs_for_query(self.graph, query)
        for cid in fused:
            src = self.children[cid]["source"]
            if src in boost:
                fused[cid] *= 1 + 0.3 * boost[src]

        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_parents(self, ranked, max_parents: int = 5) -> list[str]:
        seen, out = set(), []
        for cid, _ in ranked:
            c = self.children[cid]
            if c["parent_id"] in seen:
                continue
            seen.add(c["parent_id"])
            out.append(f"[{c['source']}]\n{self.parents[c['parent_id']]}")
        return out[:max_parents]

    @staticmethod
    def answer(query: str, contexts: list[str]) -> str:
        ctx = "\n\n---\n\n".join(contexts)
        prompt = (
            "You review contracts and policies. Use only the context below. "
            "If the answer is not there, say it is not clearly stated. "
            "Mention the source file name. Answer in the language of the question.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        )
        out = ask_llm(prompt)
        if out is None:
            return "(LLM unavailable - check API key/quota) best context found:\n" + contexts[0][:500]
        return out

    def ask(self, query: str) -> dict:
        ranked = self.retrieve(query)
        contexts = self.get_parents(ranked)
        return {
            "query": query,
            "sources": [self.children[c]["source"] for c, _ in ranked],
            "contexts": contexts,
            "answer": self.answer(query, contexts),
        }
