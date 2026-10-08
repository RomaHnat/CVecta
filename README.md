# CVecta — AI-powered CV screening & job-matching (RAG + LangGraph)

CVecta takes a job description, retrieves the most relevant CVs from a candidate
pool using vector search (RAG), scores each one against the role with an LLM, and
returns a ranked shortlist with reasoning — then runs a head-to-head comparison of
the top candidates to recommend a single best hire.

It's a demonstration of a production-shaped LLM pipeline: **retrieve-then-reason**
retrieval, structured (validated) LLM output, agentic orchestration with a real
decision loop, and an automated evaluation harness.

## Demo

> Paste a job description → get a ranked shortlist with matched/missing skills,
> a one-line verdict per candidate, and a recommended hire.

## Architecture

The pipeline is a [LangGraph](https://langchain-ai.github.io/langgraph/) state
graph. The `decide` router loops the `evaluate` node once per candidate; `rank`
then decides whether a head-to-head `compare` pass is warranted.

```mermaid
flowchart TD
    start([START]) --> extract[extract_job_requirements]
    extract --> retrieve["retrieve_candidate_pool<br/>(RAG — vector search)"]

    retrieve -->|more candidates| load["load_candidate<br/>(light text extraction)"]
    retrieve -->|none left| rank

    load -->|readable| score[score_candidate]
    load -->|PDF unreadable, OCR available| ocr["ocr_candidate<br/>(Tesseract — optional)"]
    load -->|unreadable, no OCR| skip[skip_candidate]

    ocr -->|readable| score
    ocr -->|still unreadable| skip

    score -->|more candidates| load
    score -->|all scored| rank[rank + build shortlist]
    skip -->|more candidates| load
    skip -->|all scored| rank

    rank -->|2+ shortlisted| compare["compare<br/>(head-to-head, Sonnet)"]
    rank -->|otherwise| report
    compare --> report[report]
    report --> done([END])
```

### How it works

1. **Extract** — the job description is parsed into structured requirements
   (title, required vs nice-to-have skills, min experience) using
   `with_structured_output` against a Pydantic schema.
2. **Retrieve (RAG)** — a query built from those requirements is run against a
   Chroma vector store of CV chunks. Cheap vector search narrows the pool to the
   most relevant candidates before any expensive reasoning.
3. **Load → (OCR fallback) → score** — for each candidate, a light text
   extraction runs first (`pypdf` / `python-docx` / plain text). A router checks
   whether it produced usable text. If it did, the CV is scored. If a PDF comes
   back empty (e.g. a scanned document) **and** Tesseract OCR is installed, an
   OCR node rescues it, then scoring continues. If the text can't be recovered,
   the CV is skipped and reported — the pipeline never stalls.
4. **Score** — each readable CV is scored 0–100 against the role by an LLM,
   returning matched skills, missing skills, and a short verdict as structured
   data. Every result is keyed by **filename**, so the same person can appear
   across several CVs and still be compared file-by-file.
5. **Rank & compare** — results are ranked; if two or more clear the threshold, a
   stronger model does a head-to-head reasoning pass to pick the single best hire.

### Design notes

- **Retrieve-then-reason:** vector search is cheap and fast; LLM reasoning is
  expensive. Narrowing with retrieval first keeps cost down and scales to a large
  CV pool.
- **Cost-tiered at every step:** cheap text extraction before expensive OCR; a
  fast model (Haiku) for bulk scoring, a stronger model (Sonnet) only for the
  final tie-break. Effort is spent where it changes the outcome.
- **Graceful degradation:** OCR is an optional dependency. If Tesseract isn't
  installed the graph routes around it, records which CVs it couldn't read, and
  still runs end-to-end — so the project works on a fresh clone with no extra setup.
- **Structured output everywhere:** every LLM call returns a validated Pydantic
  object, not free text — so the rest of the pipeline is ordinary, reliable Python.

## Stack

- **Python**
- **LangChain** — structured LLM output, embeddings, vector-store integration
- **LangGraph** — pipeline orchestration with decision loops and conditional branches
- **Chroma** — local persisted vector database
- **HuggingFace `all-MiniLM-L6-v2`** — local sentence embeddings (free, no API cost)
- **Pydantic** — schema-guided extraction and validation
- **pypdf / python-docx** — PDF and Word CV parsing
- **Tesseract OCR (optional)** via `pytesseract` + `PyMuPDF` — fallback for scanned PDFs
- **Streamlit** — the demo UI
- **Anthropic Claude API** — the reasoning models

## Project structure

## Project structure

```
config.py        # loads env, creates the fast (Haiku) and smart (Sonnet) LLM clients
models.py        # Pydantic schemas: CandidateProfile, JobRequirements, MatchResult, ComparisonResult
extract.py       # structured extraction of candidates and job requirements
loaders.py       # light text extraction (pdf / docx / txt) + readability check
ocr.py           # optional Tesseract OCR fallback, with graceful degradation
vectorstore.py   # builds / loads the Chroma vector store from the CV folder
matcher.py       # RAG retrieval + per-candidate LLM scoring
graph.py         # the LangGraph pipeline (the agentic core)
evaluate.py      # automated evaluation harness (positive, top-k and negative cases)
app.py           # Streamlit UI
cvs/             # candidate CVs (.txt / .pdf / .docx)
```

## Running it

```bash
# 1. Clone and enter
git clone https://github.com/<your-username>/cv-screener-rag.git
cd cv-screener-rag

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Anthropic API key
echo ANTHROPIC_API_KEY=sk-ant-... > .env

# 5. Add some CVs as .txt files into cvs/, then build the vector store
python vectorstore.py

# 6. Launch the demo
streamlit run app.py

```
### Optional: enable OCR for scanned PDFs

By default, scanned (image-only) PDFs are skipped and reported. To process them,
install the OCR extras plus the Tesseract engine:

```bash
pip install pytesseract pymupdf pillow
```

Then install the Tesseract binary (on Windows, the UB-Mannheim build). Without it,
everything still runs — scanned CVs are simply listed as "not processed".

## Evaluation

The system is checked against known-correct answers:

```bash
python evaluate.py
```

Each test case pairs a job description with the candidate(s) that *should* win.
It asserts the right candidate is recommended, appears in the top-K, and that a
deliberately mismatched candidate stays **out** of the shortlist — a negative test
proving the system can reject, not just rank. The script exits non-zero on any
failure, so it works as a regression check in CI.

## Future improvements

- **Sharper scoring** — the current scorer can cluster strong candidates near the
  top; calibrating the scoring prompt (or adding a re-ranking step) would spread
  scores out and reduce reliance on the comparison node.
- **Metadata filtering** — hard filters (location, work authorization, min years)
  before the LLM stage, so retrieval respects non-negotiables.
- **Async evaluation** — score candidates concurrently to cut latency.
- **Better PDF layout handling** — swap `pypdf` for `pdfplumber` on multi-column CVs.
- **Explainability** — surface which CV passages drove each score.
- **Deployment** — host the demo on Streamlit Community Cloud.

## Note

The CVs in this repo are synthetic, generated for demonstration. No real candidate
data is used.