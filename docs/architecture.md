# Architecture

## 1. Ingestion
1. **Parsing** - `parsing.py` converts each PDF to Markdown with Docling (keeps tables and headings).
   If Docling fails or is missing, it falls back to `pypdf`.
2. **Chunking** - `chunking.py` splits each document into *parent* chunks (800 chars, 100 overlap)
   and each parent into *child* chunks (250 chars, 50 overlap).
   Children are small, so they match queries precisely; parents are returned to the LLM so it sees full clauses.
3. **Indexing** - `indexing.py` embeds children with `all-MiniLM-L6-v2` into Qdrant (cosine) and builds a BM25 index over the same chunks.
4. **Knowledge graph** - `graph.py` builds a NetworkX graph with three node types:
   `Organization/Product`, `Clause`, `Document`.
   Entity aliases are merged into one node (e.g. "pepsico" and "pepsico egypt" -> *PepsiCo Egypt*).
   Edges: `mentioned_in`, `has_clause`, `contract_with` (parties found in a document's first 1,500 chars), `owns`.

## 2. Query time
1. **Query expansion** - the LLM rewrites the question 3 ways (falls back to keyword rewrites if no LLM).
2. **Hybrid retrieval** - for every variant, dense top-k and BM25 top-k are merged with
   Reciprocal Rank Fusion: `score = sum(1 / (60 + rank))`.
3. **Graph boost** - entities detected in the question point to their documents;
   chunks from those documents get a score multiplier of `1 + 0.3 * n_matches`.
4. **Parent expansion** - top child chunks are mapped back to unique parent chunks (max 5).
5. **Generation** - the LLM answers only from that context, names the source file, and replies in the question's language.

## 3. Why these choices
- *Hybrid search*: contracts contain exact terms (numbers, party names) where BM25 shines, plus paraphrased concepts where embeddings shine.
- *Parent/child*: precise matching without losing surrounding clause text.
- *Graph boost*: questions such as "Juhayna PepsiCo agreement" are routed to the right contract even when many contracts share similar wording.
