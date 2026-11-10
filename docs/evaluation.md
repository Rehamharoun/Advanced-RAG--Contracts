# Evaluation

The evaluation set (`data/eval_set.json`) has 8 questions, one per fact, spread across all 8 documents.

## Retrieval
| Check | Result |
|---|---|
| Expected source document appears in top results | 8 / 8 |
| Ground-truth fact appears in retrieved parent chunks | 7 / 8 |

**Known miss:** *"How is the sponsorship fee paid between PepsiCo and Vodafone?"* - the payment
schedule (40% / 30% / 30%) is not captured in the retrieved chunks, although the right document is found.
Likely cause: the schedule is formatted as a table/list that is split away from the sentence the query matches.
Ideas: larger parent size, table-aware chunking, or a reranker.

## Generation (RAGAS, judge LLM = same model as generator)
| Metric | Score |
|---|---|
| Faithfulness | 0.9375 |
| Context recall | 0.8750 |

## Fallback LLM-judge
Faithfulness 8 / 8; correctness against ground truth 7 / 8.

## DSPy
`BootstrapFewShot` with 2 bootstrapped + 2 labelled demos: base 7 / 8, tuned 7 / 8.
No gain on this small set; the optimiser is included to show the workflow.

## Reproduce
```bash
python scripts/run_pipeline.py --eval --dspy
```
Numbers may vary slightly between runs because the LLM is non-deterministic.
