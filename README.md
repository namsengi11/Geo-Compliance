# Geo-Compliance

# Getting Started

## 0. Clone

```bash
git clone https://github.com/namsengi11/Geo-Compliance.git
cd geo-compliance
```

## 1. Python environment

Use Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

## 2. Install dependencies

> PyTorch wheels are CUDA/OS specific. Install Torch first for your machine from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/), then install the rest.

```bash
# Example (adjust for your CUDA/CPU):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

pip install -r requirements.txt
```

## 3. Prepare the vector DB (one-time ingest)

Place regulation files under `./regulations/` and list them in `texts-available.csv` as:

```
<Region>,<Filename>
California,Protecting-Our-Kids-from-Social-Media-Addiction-Act.html
European Union,EU-Digital-Service-Act.html
Utah,Utah Social Media Regulation Act.pdf
...
```

Then build the Chroma store:

```bash
python document_manager.py
```

This creates a single Chroma database with each chunk tagged by `metadata["region"] = <Region>`.

---

## 4. Run with Gemini (cloud)

1. Enable the Generative Language API for your Google account/project.
2. Set the API key:

```bash
export GEMINI_API_KEY="YOUR_KEY"    # Windows PowerShell: $env:GEMINI_API_KEY="YOUR_KEY"
```

3. CLI — Retrieval + JSON output:

```bash
python main.py --query "Trial run of video replies in EEA only. GH manages exposure; BB baselines feedback." --model gemini -k 5
```

4. UI — Streamlit demo:

```bash
streamlit run demo_app.py
```

* Enter a feature name and description, choose model = `gemini-2.5-flash` or `gemini-2.5-pro`, set Top-K, then Evaluate.
* The app displays the strict JSON, a download button, and a collapsible “History” panel. Each run upserts to your CSV log (default `sample_data_response.csv`).

---

## 5. Run locally with Llama (offline)

Model: `meta-llama/Meta-Llama-3-8B-Instruct` (Hugging Face)

1. Accept the model license on Hugging Face and log in locally:

```bash
huggingface-cli login
```

2. (Optional) Ensure the model files are accessible (HF cache or local path). If you use quantized weights, install the right extras (e.g., `bitsandbytes`) and adjust your local LLM loader accordingly.

3. CLI — Retrieval + JSON output with the local model:

```bash
python main.py --query "To comply with the Utah Social Media Regulation Act, we will enforce minors’ curfew login windows." --model local -k 5
```

Notes:

* An 8B model benefits from a GPU with ample VRAM; otherwise use CPU or a lighter/quantized variant.
* The pipeline, prompts, and output schema are the same as the Gemini path; only the backend LLM differs.

---

## 6. Common issues

* **No retrieved documents / empty response**
  Ensure `document_manager.py` ran successfully and your `texts-available.csv` paths match files under `./regulations/`. Check that your query’s region is present; Top-K can be lowered to improve precision.

* **Gemini error: API key invalid**
  Verify `GEMINI_API_KEY` is set in the same shell that runs `python main.py` or `streamlit run demo_app.py`.

* **Partial JSON response**
  Reduce Top-K (e.g., 3–5), keep feature descriptions concise, and ensure your token limits are adequate in the sidebar. The app validates and parses JSON; if you see truncation, simplify the query or retrieved context.

* **Torch / CUDA install problems**
  Install PyTorch from the official wheel index matching your CUDA/OS, then `pip install -r requirements.txt`.

---

That’s it. After ingestion, you can use either the CLI or the Streamlit app. The same RAG + structured-JSON evaluation runs in both Gemini and local Llama modes.



## Project Features & Functionality

Our project is a Geo-Regulation Compliance Assistant that evaluates product features or code changes against region-specific legal frameworks (for example, Utah Social Media Regulation Act, California SB-976, and the EU Digital Services Act).

We implement a Retrieval-Augmented Generation (RAG) pipeline:

1. Regulations are ingested from PDF/HTML/DOCX, chunked with page indices, embedded, and stored in Chroma.
2. A feature description is provided by the user.
3. A lightweight region classifier selects the relevant jurisdiction(s).
4. A glossary expansion step resolves internal acronyms (PF, GH, ASL, BB, EchoTrace, etc.).
5. The retriever fetches the top-k jurisdiction-specific clauses.
6. The LLM generates a strict, machine-readable JSON object that includes a compliance flag, issues, reasoning, and evidence quoted from the source.

Additional capabilities

* Developer document evaluator: extracts features from PRDs/dev docs for downstream compliance checks.
* Code-change evaluator: summarizes feature-level impacts from diffs and maps them to regulatory requirements.
* Streamlit demo app: interactive UI that runs the same pipeline, displays JSON output, and supports run history logging.
* CSV logging and history panel: every run is upserted to a CSV and viewable in a collapsible, scrollable log for traceability.

## Models

* Primary cloud model: Google Gemini (gemini-2.5-flash / gemini-2.5-pro) for structured JSON responses.
* Local model: meta-llama/Meta-Llama-3-8B-Instruct (quantized for local inference) used as an on-prem fallback and for offline development. The same prompt format and RAG pipeline are used, with adjustments for chat templates and output parsing.

## Development Tools

* Python 3.10
* Streamlit for the demo UI
* Virtualenv/venv for environment management
* Git for version control
* Optional local serving: Hugging Face Transformers and quantization utilities; vLLM experiments were conducted but not required in the current demo
* Command-line utilities and simple batch runners for offline tests

## APIs Used

* Google Generative Language API (Gemini) via langchain-google-genai for structured output and schema-constrained responses.

Note: The local LLM path does not call external APIs; it uses local inference through the transformers stack.

## Assets Used

* Regulation source files stored under `regulations/` (e.g., Utah Social Media Regulation Act PDF, California SB-976 HTML, EU DSA HTML).
* Internal terminology glossary (PF, GH, ASL, BB, EchoTrace, etc.) integrated as a searchable resource to normalize product acronyms.
* Example PRDs and developer documents used for feature-extraction tests.

## Libraries Used

* LangChain, langchain-community, langchain-google-genai for LLM orchestration, prompts, and chains
* ChromaDB for vector storage and retrieval
* sentence-transformers (all-MiniLM-L6-v2) for embeddings
* Hugging Face Transformers for local LLM inference
* Streamlit for the UI
* Pandas/CSV for result storage and history logging
* Pydantic/json for schema handling and robust parsing

## Problem Statement

Product teams must ship features that comply with a patchwork of regional regulations. Manual checks are slow, inconsistent, and hard to audit. The goal is to automate compliance discovery by:

1. Mapping a plain-language feature description or code change to applicable jurisdictions.
2. Retrieving the most relevant regulatory clauses.
3. Producing a structured, auditable JSON response with issues, reasoning, and verbatim evidence from the law.
4. Enabling the same checks to run in an automated gate (pre-commit or pre-PR) for continuous compliance.

## Additional Datasets (Beyond the Problem Statement)

* Internal Terminology Glossary: A small, curated dataset of product acronyms and operational terms used to disambiguate feature descriptions before retrieval.
* Sample PRD/Dev-Doc Corpus: A synthetic set of product requirement documents used to test the feature-extraction and developer-doc evaluation pipelines.

## Notes on Integration and CI/CD

* The evaluation pipeline is callable from a CLI or service endpoint and can run in a pre-commit or pre-PR job.
* When issues are detected, the pipeline returns a strictly structured JSON artifact that can be attached to the PR for legal review; when insufficient context exists, a deterministic fallback JSON is emitted.
* The Streamlit demo uses the exact same backend logic, ensuring parity between interactive demos and CI checks.

---




