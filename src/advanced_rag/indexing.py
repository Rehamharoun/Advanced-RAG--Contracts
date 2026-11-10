"""Dense (Qdrant) + sparse (BM25) indexes over child chunks."""
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from . import config


@dataclass
class Index:
    embed_model: SentenceTransformer
    qdrant: QdrantClient
    bm25: BM25Okapi


def build_index(children: list[dict]) -> Index:
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    texts = [c["text"] for c in children]
    vectors = embed_model.encode(texts, batch_size=32).tolist()

    qdrant = QdrantClient(":memory:")
    if qdrant.collection_exists(config.COLLECTION):
        qdrant.delete_collection(config.COLLECTION)
    qdrant.create_collection(
        config.COLLECTION,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )

    for s in range(0, len(children), 200):
        points = [
            PointStruct(
                id=i,
                vector=vectors[i],
                payload={"source": children[i]["source"],
                         "parent_id": children[i]["parent_id"],
                         "text": texts[i]},
            )
            for i in range(s, min(s + 200, len(children)))
        ]
        qdrant.upsert(config.COLLECTION, points=points)
    print("stored in qdrant:", qdrant.count(config.COLLECTION).count)

    bm25 = BM25Okapi([config.tokenize(t) for t in texts])
    return Index(embed_model=embed_model, qdrant=qdrant, bm25=bm25)
