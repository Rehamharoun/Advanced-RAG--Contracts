# Data

## PDFs
Place the contract / policy PDFs in `data/pdfs/`. They are ignored by git (see `.gitignore`)
so that no confidential or licensed documents are pushed by mistake.

The evaluation set expects file names that start with these prefixes:

| Prefix | Document |
|---|---|
| `01_` | Meta Privacy Policy (sample) |
| `02_` | WhatsApp Business Terms (sample) |
| `03_` | Service Agreement - Social Integration |
| `04_` | Juhayna - PepsiCo Joint Distribution Agreement |
| `05_` | Vodafone Egypt Managed Services Agreement |
| `06_` | PepsiCo - Vodafone Sponsorship Media Agreement |
| `07_` | Juhayna Cold Chain Logistics Agreement |
| `08_` | Juhayna Supplier Framework and Code of Conduct |

## eval_set.json
Each item has: `q` (question), `src` (expected file prefix), `key` (fact that must appear
in the retrieved context / answer), and `truth` (reference answer used by RAGAS).
