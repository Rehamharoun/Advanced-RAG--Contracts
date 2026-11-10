# Setup

## Requirements
- Python 3.10+
- A Groq and/or Gemini API key (free tiers work; the code waits and retries on rate limits)

## Local
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your keys
```
Put the PDFs in `data/pdfs/` and run:
```bash
python scripts/run_pipeline.py --query "Your question here"
```

## Tests
```bash
pip install pytest
pytest
```

## Security
- Never commit `.env` or hard-code keys in source files. If a key was ever exposed, revoke and regenerate it.
