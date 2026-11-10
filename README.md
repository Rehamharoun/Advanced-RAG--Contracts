# Advanced RAG for Contract Q&A

A proof-of-concept Retrieval-Augmented Generation (RAG) system that answers questions
over a collection of legal contracts and policies, always citing the source document.

It combines **hybrid retrieval**, **parent/child chunking**, a lightweight **knowledge graph**,
and a measured **evaluation loop** (RAGAS + DSPy).

## Highlights

| Stage | Technique | Tooling |
|---|---|---|
| Parsing | PDF to Markdown (keeps tables), pypdf fallback | Docling, pypdf |
| Chunking | Parent / child chunks (800 / 250 chars) | LangChain text splitters |
| Dense search | Sentence embeddings + cosine similarity | `all-MiniLM-L6-v2`, Qdrant |
| Sparse search | Keyword matching | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) of dense + BM25 | custom |
| Query expansion | LLM rewrites the question 3 ways | Groq / Gemini |
| Graph boost | Entities in the query boost related documents | NetworkX |
| Generation | Answer only from retrieved context, cite file name | Groq `gpt-oss-120b` / Gemini 2.5 Flash |
| Evaluation | Faithfulness, context recall, LLM-judge, DSPy tuning | RAGAS, DSPy |

## Architecture

```
PDFs -> Docling -> parent/child chunks -> Qdrant (dense) + BM25 (sparse)
                                   \-> Knowledge graph (entities, clauses, documents)

Question -> query expansion -> hybrid retrieval (RRF) -> graph boost
         -> parent chunks as context -> LLM answer with sources
```

More detail in [`docs/architecture.md`](docs/architecture.md).

## Results (8-question evaluation set)

| Metric | Score |
|---|---|
| Correct source document in top results | 8 / 8 |
| Ground-truth fact present in retrieved context | 7 / 8 |
| Answer contains the key fact | 7 / 8 |
| RAGAS faithfulness | 0.94 |
| RAGAS context recall | 0.88 |
| LLM-judge faithfulness / correctness | 8 / 8 and 7 / 8 |
| DSPy BootstrapFewShot (base vs tuned) | 7 / 8 vs 7 / 8 |

Details and the known failure case are in [`docs/evaluation.md`](docs/evaluation.md).

## Project structure

```
Advanced-RAG--Contracts/
├── README.md
├── LICENSE
├── app.py                  # Streamlit chat demo
├── requirements.txt
├── .env.example            # template for API keys (copy to .env)
├── .gitignore
├── data/
│   ├── README.md           # how to add the contract PDFs
│   ├── eval_set.json       # 8 questions with ground truth
│   └── pdfs/               # put your PDFs here (not tracked by git)
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   └── setup.md
├── scripts/
│   └── run_pipeline.py     # command-line entry point
├── src/advanced_rag/       # reusable package
│   ├── config.py  llm.py  parsing.py  chunking.py
│   ├── indexing.py  graph.py  retrieval.py
│   ├── pipeline.py  evaluation.py
└── tests/
    └── test_utils.py
```

## Quick start

```bash
git clone https://github.com/Rehamharoun/Advanced-RAG--Contracts.git
cd Advanced-RAG--Contracts

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # Windows: copy .env.example .env
# edit .env and add GROQ_API_KEY and/or GEMINI_API_KEY

# add the contract PDFs to data/pdfs/  (see data/README.md)

python scripts/run_pipeline.py --query "What is the liability cap in the social media integration agreement?"
python scripts/run_pipeline.py --eval --graph
```

Launch the chat demo UI:

```bash
streamlit run app.py
```

Full setup notes: [`docs/setup.md`](docs/setup.md).

## Configuration

All settings come from environment variables (`.env`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | - | Groq API key |
| `GEMINI_API_KEY` | - | Gemini API key |
| `USE_GROQ` | `true` | `true` = Groq, `false` = Gemini |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model name |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `PDF_DIR` | `data/pdfs` | Folder containing the PDFs |

## Limitations and next steps

- Entity list and clause keywords in the knowledge graph are hand-written; automatic entity extraction would scale better.
- The evaluation set is small (8 questions); it shows the approach works but is not a statistically strong benchmark.
- Qdrant runs in memory; switch to a persistent / server instance for production.
- English embedding model; Arabic contracts would need a multilingual model.

## Author

**Reham Haroun** - Machine Learning Engineer  
[GitHub](https://github.com/Rehamharoun) · [LinkedIn](https://linkedin.com/in/rehamharoun)

Licensed under the MIT License.
