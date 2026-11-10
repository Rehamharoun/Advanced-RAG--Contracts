"""Evaluation: retrieval check, RAGAS, LLM-judge fallback, optional DSPy tuning."""
import json
import re
import sys
import time
import types
from pathlib import Path

from . import config
from .llm import ask_llm

EVAL_SET_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_set.json"


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fact_in(text: str, key: str) -> bool:
    """True if key (or all numbers in it) appear in text - tolerant to wording."""
    t, k = config.norm(text), config.norm(key)
    if k in t:
        return True
    nums = re.findall(r"\d+", k)
    return bool(nums) and all(n in t for n in nums)


def retrieval_check(retriever, eval_set: list[dict]) -> dict:
    src_hits = key_hits = 0
    for item in eval_set:
        ranked = retriever.retrieve(item["q"])
        sources = [retriever.children[c]["source"] for c, _ in ranked]
        item["contexts"] = retriever.get_parents(ranked)
        src_ok = any(s.startswith(item["src"]) for s in sources)
        key_ok = any(config.norm(item["key"]) in config.norm(c) for c in item["contexts"])
        src_hits += src_ok
        key_hits += key_ok
        print(f"{'ok ' if src_ok else 'MISS'} src | {'ok ' if key_ok else 'MISS'} fact | {item['q']}")
    n = len(eval_set)
    print(f"source hit: {src_hits}/{n}  fact in context: {key_hits}/{n}")
    return {"source_hit": src_hits, "fact_in_context": key_hits, "total": n}


def generate_answers(retriever, eval_set: list[dict], sleep: int = 4) -> bool:
    for item in eval_set:
        item["response"] = retriever.answer(item["q"], item["contexts"])
        print("answered:", item["q"][:60])
        time.sleep(sleep)  # stay under free-tier rate limits
    ok = not any(i["response"].startswith("(LLM unavailable") for i in eval_set)
    if ok:
        print("answer contains fact:", sum(fact_in(i["response"], i["key"]) for i in eval_set), "/", len(eval_set))
    return ok


def _patch_vertexai() -> None:
    """Work around a langchain_community import issue triggered by ragas."""
    m = types.ModuleType("langchain_community.chat_models.vertexai")
    m.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = m
    try:
        import langchain_community.llms as L

        L.VertexAI = type("VertexAI", (), {})
    except Exception as e:
        print("llms patch:", e)


def run_ragas(eval_set: list[dict]):
    """Faithfulness + context recall. Returns the result or None on failure."""
    _patch_vertexai()
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextRecall
        from ragas.run_config import RunConfig

        if config.USE_GROQ:
            from langchain_groq import ChatGroq

            chat = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY)
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI

            chat = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL,
                                          google_api_key=config.GEMINI_API_KEY)

        rows = [{"user_input": i["q"], "retrieved_contexts": i["contexts"],
                 "response": i["response"], "reference": i["truth"]} for i in eval_set]
        result = evaluate(
            EvaluationDataset.from_list(rows),
            metrics=[Faithfulness(), LLMContextRecall()],
            llm=LangchainLLMWrapper(chat),
            run_config=RunConfig(max_workers=1, timeout=120, max_retries=3),
            show_progress=True,
        )
        print(result)
        return result
    except Exception as e:
        print("ragas failed:", repr(e)[:400])
        return None


def llm_judge(eval_set: list[dict], sleep: int = 4) -> dict:
    """Simple fallback when RAGAS is unavailable."""
    f_scores, c_scores = [], []
    for i in eval_set:
        ctx = "\n\n".join(i["contexts"])[:6000]
        f = ask_llm(f"Context:\n{ctx}\n\nAnswer:\n{i['response']}\n\n"
                    "Is every claim in the answer supported by the context? Reply only YES or NO.")
        c = ask_llm(f"Reference answer: {i['truth']}\nCandidate answer: {i['response']}\n\n"
                    "Does the candidate contain the same key fact as the reference? Reply only YES or NO.")
        f_scores.append(1 if f and f.strip().upper().startswith("YES") else 0)
        c_scores.append(1 if c and c.strip().upper().startswith("YES") else 0)
        time.sleep(sleep)
    n = len(eval_set)
    print(f"faithfulness (LLM judge): {sum(f_scores)}/{n}")
    print(f"correctness vs ground truth (LLM judge): {sum(c_scores)}/{n}")
    return {"faithfulness": sum(f_scores), "correctness": sum(c_scores), "total": n}


def run_dspy(eval_set: list[dict]) -> None:
    """Few-shot prompt optimisation with DSPy BootstrapFewShot."""
    try:
        import dspy

        model_name = f"groq/{config.GROQ_MODEL}" if config.USE_GROQ else f"gemini/{config.GEMINI_MODEL}"
        dspy.configure(lm=dspy.LM(model_name,
                                  api_key=config.GROQ_API_KEY if config.USE_GROQ else config.GEMINI_API_KEY,
                                  num_retries=5, timeout=60))

        class ContractQA(dspy.Signature):
            """Answer the contract question using only the context."""
            context = dspy.InputField()
            question = dspy.InputField()
            answer = dspy.OutputField()

        def metric(example, pred, trace=None):
            return fact_in(pred.answer, example.key)

        trainset = [dspy.Example(context="\n\n".join(i["contexts"]), question=i["q"], key=i["key"])
                    .with_inputs("context", "question") for i in eval_set]
        base = dspy.ChainOfThought(ContractQA)
        tuned = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=2,
                                      max_labeled_demos=2).compile(base, trainset=trainset)

        for name, prog in [("base", base), ("tuned", tuned)]:
            ok = 0
            for ex in trainset:
                ok += bool(metric(ex, prog(context=ex.context, question=ex.question)))
                time.sleep(3)
            print(name, ok, "/", len(trainset))
    except Exception as e:
        print("dspy failed:", repr(e)[:400])
