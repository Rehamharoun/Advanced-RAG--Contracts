"""Thin LLM wrapper (Groq or Gemini) with retry on rate limits."""
import time

from . import config

_clients: dict = {}


def get_client(kind: str):
    if kind not in _clients:
        if kind == "groq":
            from groq import Groq

            _clients[kind] = Groq(api_key=config.GROQ_API_KEY)
        else:
            from google import genai

            _clients[kind] = genai.Client(api_key=config.GEMINI_API_KEY)
    return _clients[kind]


def ask_llm(prompt: str, retries: int = 4):
    """Return the model reply, or None if no key / persistent failure."""
    for attempt in range(retries):
        try:
            if config.USE_GROQ and config.GROQ_API_KEY:
                r = get_client("groq").chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2048,
                )
                return r.choices[0].message.content
            if not config.USE_GROQ and config.GEMINI_API_KEY:
                r = get_client("gemini").models.generate_content(
                    model=config.GEMINI_MODEL, contents=prompt
                )
                return r.text
            return None
        except Exception as e:
            msg = str(e)
            rate_limited = any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "quota", "rate"))
            if rate_limited and attempt < retries - 1:
                wait = 15 * (attempt + 1)
                print(f"rate limit, waiting {wait}s...")
                time.sleep(wait)
                continue
            print("llm error:", msg[:300])
            return None
    return None
