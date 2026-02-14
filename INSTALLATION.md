# Installation Guide

Step-by-step instructions to set up and run the Fraud Q&A Agent Chatbot from scratch.

---

## Prerequisites

| Requirement | Version | Check Command |
|---|---|---|
| **Python** | 3.11 or higher | `python --version` |
| **pip** | Latest | `pip --version` |
| **Git** | Any | `git --version` |
| **OpenAI API Key** | - | [Get one here](https://platform.openai.com/api-keys) |

> **Cost Note**: Running the chatbot costs approximately **$0.001–0.002 per question** (GPT-4o-mini). The one-time data ingestion costs ~$0.003 for embedding 184 document chunks.

---

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd aiengineer-assesment
```

---

## Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment:

| OS | Command |
|---|---|
| **macOS / Linux** | `source .venv/bin/activate` |
| **Windows (CMD)** | `.venv\Scripts\activate.bat` |
| **Windows (PowerShell)** | `.venv\Scripts\Activate.ps1` |

You should see `(.venv)` in your terminal prompt.

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the following packages:

| Package | Purpose |
|---|---|
| `pydantic-ai` | Agent framework with tool routing |
| `openai` | GPT-4o-mini chat + text-embedding-3-small |
| `streamlit` | Web-based chat UI |
| `duckdb` | Fast analytical SQL database |
| `faiss-cpu` | FAISS vector similarity search |
| `pymupdf` | PDF text extraction |
| `langchain-text-splitters` | Text chunking for RAG |
| `python-dotenv` | Environment variable loading |
| `plotly` | Interactive charts |
| `numpy` | Numerical operations |
| `tiktoken` | Token counting |
| `pytest` | Test runner |

---

## Step 4: Configure the API Key

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-api-key-here
```

> **Security**: The `.env` file is listed in `.gitignore` and will not be committed.

### Optional Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | Language model for chat and scoring |
| `CHUNKING_MODE` | `semantic` | Chunking strategy: `semantic` or `fixed` |
| `CHUNK_SIZE` | `1000` | Fixed chunking: characters per chunk |
| `CHUNK_OVERLAP` | `200` | Fixed chunking: overlap between chunks |
| `SEMANTIC_MIN_CHUNK` | `100` | Semantic chunking: minimum chunk size |
| `SEMANTIC_MAX_CHUNK` | `1500` | Semantic chunking: maximum chunk size |

---

## Step 5: Download the Dataset

Download the Fraud Detection dataset from Kaggle:

1. Go to [https://www.kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
2. Click **Download** (requires a Kaggle account)
3. Extract the ZIP file. You should have `fraudTrain.csv` and `fraudTest.csv`
4. Place both CSV files in the **project root directory**:

```
aiengineer-assesment/
├── fraudTrain.csv        ← Place here
├── fraudTest.csv         ← Place here
├── app.py
└── ...
```

> **Note**: The CSV files are ~150MB and ~350MB respectively.

---

## Step 6: Verify PDF Documents

The project uses two PDF research documents for RAG. Verify they exist in the project root:

```
aiengineer-assesment/
├── Bhatla.pdf                              ← Should already be here
├── EBA_ECB 2024 Report on Payment Fraud.pdf ← Should already be here
└── ...
```

These files are included in the repository. If missing, refer to the assessment instructions.

---

## Step 7: Run Data Ingestion

This one-time step processes the raw data into optimized formats:

```bash
python scripts/ingest.py
```

**What this does:**

1. **CSV → DuckDB** (`data/processed/fraud.duckdb`)
   - Loads `fraudTrain.csv` + `fraudTest.csv` into a single `transactions` table
   - Creates indexes for efficient querying
   - Result: ~1.85M rows

2. **PDF → FAISS** (`data/processed/faiss_index.bin` + `chunks.pkl`)
   - Extracts text from both PDFs
   - Splits into chunks using the configured chunking strategy
   - Generates embeddings via `text-embedding-3-small`
   - Builds a FAISS similarity search index
   - Result: ~184 chunks

**Expected output:**

```
12:00:00 [INFO] ingest: === FRAUD Q&A CHATBOT - DATA INGESTION ===
12:00:00 [INFO] ingest: [1/2] Loading CSV files into DuckDB...
12:00:05 [INFO] ingest: [1/2] Complete: 1,852,394 transactions loaded
12:00:05 [INFO] ingest: [2/2] Processing PDFs into FAISS...
12:00:12 [INFO] ingest: [2/2] Complete: 184 chunks indexed
12:00:12 [INFO] ingest: === INGESTION COMPLETE ===
```

**Expected files created:**

```
data/
└── processed/
    ├── fraud.duckdb           # ~250MB DuckDB database
    ├── faiss_index.bin        # FAISS vector index
    └── chunks.pkl             # Serialized chunk metadata
```

---

## Step 8: Run the Application

```bash
streamlit run app.py
```

The chatbot will open in your browser at **http://localhost:8501**.

---

## Step 9: Verify It Works

Try one of these example questions:

| Type | Question |
|---|---|
| **SQL** | "How does the monthly fraud rate fluctuate over the two-year period?" |
| **SQL** | "Which merchant categories exhibit the highest incidence of fraudulent transactions?" |
| **RAG** | "What are the primary methods by which credit card fraud is committed?" |
| **Hybrid** | "What are the core components of an effective fraud detection system?" |

Every response should include:
- ✅ An answer
- ✅ SQL query / data table (for SQL questions) or source citations (for RAG questions)
- ✅ Quality score badge (green = high quality)

---

## Running Tests

```bash
pytest tests/ -v
```

All **30 tests** should pass. Tests use mocking and do not require an OpenAI API key or data files.


## Quick Reference

| Command | Description |
|---|---|
| `python scripts/ingest.py` | Process CSV + PDF data (run once) |
| `streamlit run app.py` | Start the chatbot |
| `pytest tests/ -v` | Run all tests |
| `cat .env.example` | See required environment variables |
