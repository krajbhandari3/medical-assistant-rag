# Medical Assistant: RAG-Based Clinical Q&A

A Retrieval-Augmented Generation (RAG) prototype that answers clinical questions using only the **Merck Manual**, then measures whether each answer is grounded in the source and relevant to the question.

*UT Austin – Post Graduate Program in Artificial Intelligence & Machine Learning*

---

## Business Problem

Healthcare professionals face information overload and need accurate answers fast, especially in emergencies. General-purpose LLMs answer fluently but without sources, and they can be confidently wrong.

**Objective:** build and evaluate a RAG system that answers clinical questions from a trusted medical reference, with every answer traceable to a source page.

## Approach

| Stage | What was done |
|---|---|
| 1. Baseline LLM | `gpt-4o-mini` answers 5 clinical questions from memory |
| 2. Prompt engineering | Clinical system prompt (7 rules) + 5 parameter combinations |
| 3. RAG data prep | Load, clean, chunk and embed the manual; store vectors in ChromaDB |
| 4. RAG + fine-tuning | Answer only from retrieved context; tune `k`, `temperature`, `top_p`, `max_tokens` |
| 5. Evaluation | `gpt-4o` as an independent judge scores groundedness and relevance (1–5) |

**Questions tested:** sepsis protocol in the ICU · appendicitis symptoms and treatment · sudden patchy hair loss · traumatic brain injury · leg fracture on a hiking trip.

## RAG Pipeline

| Component | Choice |
|---|---|
| Source | Merck Manual PDF: 4,114 pages → 4,112 after cleaning |
| Cleaning | Removed a watermark on every page and table-of-contents dot leaders (5.9% of text) |
| Chunking | `RecursiveCharacterTextSplitter` (tiktoken `cl100k_base`), `chunk_size=400`, `chunk_overlap=50` → **10,430 chunks** |
| Embeddings | `thenlper/gte-large` (1,024-dim, normalised, GPU) |
| Vector store | ChromaDB, persisted to disk |
| Retriever | Similarity search, `k=3` (default) |
| Generator / Judge | `gpt-4o-mini` / `gpt-4o` via the OpenAI API |

## Key Results

| Question | Groundedness | Relevance |
|---|:---:|:---:|
| Sepsis | 5 | 5 |
| Appendicitis | 5 | 5 |
| Hair loss | 5 | 5 |
| Brain injury | 5 | 5 |
| Leg fracture (k=6) | 4 | 5 |

## Key Insights

1. **RAG made answers safer and traceable.** The baseline LLM recommended corticosteroids for traumatic brain injury (a harmful practice); the RAG answer did not.
2. **Prompt engineering improves structure; RAG improves trust.** The system prompt produced clinician-ready answers, but only RAG made them verifiable.
3. **Retrieval quality drives answer quality.** Raising `k` from 3 to 6 fixed the fracture answer, which had been built on stress-fracture passages. Temperature, top_p and max_tokens mostly changed wording and length.
4. **Low temperature (0–0.2) suits clinical use.** `max_tokens` below ~400 truncated answers without warning.
5. **LLM-as-a-judge is useful but not sufficient.** It gave 5/5 relevance to mis-targeted answers, so human review is still needed.

## Recommendations

- Position the tool as **decision support**, showing source page numbers with every answer.
- **Improve retrieval** before scaling: `k=5–6`, MMR search, and metadata filters (e.g., adult vs. paediatric).
- **Keep the knowledge base current**, since grounded answers inherit the source's content.
- Pair automated scores with **periodic clinician review**.

## Repository Contents

| File | Description |
|---|---|
| `Medical_Assistant_RAG.ipynb` | Full notebook: code, outputs and observations |
| `Medical_Assistant_RAG.html` | Rendered notebook for quick viewing |
| `Medical_Assistant_Presentation.pptx` | Summary presentation of findings |
| `requirements.txt` | Python dependencies |

> **Note:** The Merck Manual PDF and the API key file (`config.json`) are not included. To run the notebook, provide your own copy of the manual and an OpenAI API key.

## How to Run

```bash
pip install -r requirements.txt
```

1. Place the Merck Manual PDF in the project folder as `medical_diagnosis_manual.pdf`.
2. Create `config.json` with your OpenAI credentials:
   ```json
   {"OPENAI_API_KEY": "your-key", "OPENAI_API_BASE": "https://api.openai.com/v1"}
   ```
3. Open the notebook and run the cells in order. The first run builds the vector store (a GPU is recommended); later runs reload it in seconds.

## Tech Stack

Python · LangChain · ChromaDB · sentence-transformers · PyMuPDF · tiktoken · OpenAI API · Jupyter

---

**Kshitij Rajbhandari** · [GitHub](https://github.com/krajbhandari3)
