import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advanced_rag.config import norm, tokenize  # noqa: E402
from advanced_rag.evaluation import fact_in, load_eval_set  # noqa: E402


def test_tokenize():
    assert tokenize("Liability Cap, 12 months!") == ["liability", "cap", "12", "months"]


def test_norm():
    assert norm("  Hello   WORLD ") == "hello world"


def test_fact_in_exact_and_numeric():
    assert fact_in("Notice period is 120 days.", "one hundred twenty (120) days")
    assert not fact_in("Notice period is 30 days.", "120 days")


def test_eval_set_schema():
    items = load_eval_set()
    assert len(items) == 8
    assert all({"q", "src", "key", "truth"} <= set(i) for i in items)
