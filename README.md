# 🔍 Fraud Q&A Agent Chatbot

An intelligent chatbot that answers questions about credit card fraud using both **tabular transaction data** and **research documents**. Built for the Mekari AI Engineer assessment.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                         │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────────┐ │
│  │  Chat Panel   │  │  Results  │  │  Quality Score   │ │
│  │  (messages)   │  │ (table/   │  │  (faithfulness,  │ │
│  │               │  │  chart)   │  │   relevance,     │ │
│  │               │  │           │  │   confidence)    │ │
│  └──────┬────────┘  └─────▲─────┘  └──────▲───────────┘ │
└─────────┼────────────────┼───────────────┼──────────────┘
          │                │               │
          ▼                │               │
┌─────────────────────────────────────────────────────────┐
│              Router Agent (PydanticAI + OpenAI)          │
│       Classifies query intent → selects tool(s)         │
│                                                         │
│  ┌───────────────┐          ┌───────────────┐           │
│  │   SQL Tool     │          │   RAG Tool    │           │
│  │ (Text-to-SQL)  │          │ (PDF Search)  │           │
│  │                │          │               │           │
│  │ OpenAI generates│         │ FAISS search  │           │
│  │ DuckDB query   │          │ → OpenAI      │           │
│  │ → execute      │          │   summarize   │           │
│  │ → format result│          │ → cite source │           │
│  └───────┬────────┘          └───────┬───────┘           │
│          │                           │                   │
│          ▼                           ▼                   │
│  ┌───────────────┐          ┌───────────────┐           │
│  │    DuckDB      │          │    FAISS      │           │
│  │ (1.85M rows)   │          │  (PDF chunks  │           │
│  │ fraud data     │          │  + embeddings)│           │
│  └────────────────┘          └───────────────┘           │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Quality Scoring Module                  │   │
│  │  Faithfulness (LLM-judge) + Relevance (cosine)   │   │
│  │  + Confidence (retrieval scores / SQL success)    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| **Agent Framework** | PydanticAI | Type-safe, tool registration, dependency injection |
| **LLM** | OpenAI GPT-4o-mini | Cost-effective, excellent at Text-to-SQL and RAG |
| **Embeddings** | OpenAI text-embedding-3-small | 1536 dims, high quality |
| **Vector Store** | FAISS (faiss-cpu) | Fast similarity search, no server needed |
| **SQL Database** | DuckDB | Blazing fast analytical queries on CSV data |
| **PDF Parsing** | PyMuPDF (fitz) | Fast, accurate text extraction |
| **UI** | Streamlit | Built-in chat components, easy data display |
| **Charts** | Plotly | Interactive dark-theme visualizations |

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key

### Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd aiengineer-assesment

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 5. Place data files
# Put fraudTrain.csv, fraudTest.csv, Bhatla.pdf, and
# EBA_ECB 2024 Report on Payment Fraud.pdf in data/raw/

# 6. Run data ingestion (one-time)
python scripts/ingest.py

# 7. Start the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Data Sources

### Transaction Dataset
- **Source**: [Kaggle Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **Records**: ~1,852,394 transactions
- **Period**: January 2019 – December 2020
- **Fraud rate**: ~0.6% (9,651 fraudulent transactions)
- **Categories**: 14 merchant categories

### Research Documents
- **Bhatla et al.** — "Understanding Credit Card Frauds" (2003, 15 pages)
- **EBA/ECB** — "2024 Report on Payment Fraud" (August 2024, 35 pages)

## Features

### 🔍 Intelligent Query Routing
The agent automatically determines whether to use SQL (data questions) or RAG (document questions), or both for mixed queries.

### 📊 Auto-Visualization
- Time-series data → interactive line charts
- Categorical data → bar charts
- All results shown in sortable data tables

### 📋 Quality Scoring
Every response is scored on three dimensions:
- **Faithfulness** (0-1): Is the answer supported by the evidence? (LLM-as-judge)
- **Relevance** (0-1): How similar is the answer to the question? (embedding cosine similarity)
- **Confidence** (0-1): How confident are the retrieval results? (FAISS scores / SQL success)
- **Overall** = 50% faithfulness + 30% relevance + 20% confidence

### 🛡️ Safety & Error Handling
- SQL injection prevention (SELECT-only queries)
- PII masking (credit card numbers, names masked in output)
- Self-correcting SQL (retries with error feedback)
- Graceful error handling throughout

## Example Questions

| Question | Source | Expected Answer |
|---|---|---|
| "How does the monthly fraud rate fluctuate?" | SQL | Monthly rates with line chart |
| "Which categories have highest fraud?" | SQL | Category ranking with bar chart |
| "What are the primary fraud methods?" | RAG (Bhatla) | Lost/stolen 48%, Identity theft 15%, etc. |
| "Core components of fraud detection system?" | RAG (Bhatla) | Manual review, AVS, CVM, neural nets, etc. |
| "Fraud rates outside the EEA?" | RAG (EBA/ECB) | ~10x higher for SCA-authenticated payments |
| "Cross-border fraud share in H1 2023?" | RAG (EBA/ECB) | 71% of total card fraud value |

## Project Structure

```
├── app.py                     # Streamlit entry point
├── requirements.txt           # Python dependencies
├── .env.example               # API key template
├── scripts/
│   └── ingest.py              # One-time data ingestion
├── src/
│   ├── agent/
│   │   ├── prompts.py         # All centralized prompts
│   │   ├── router.py          # PydanticAI agent + routing
│   │   ├── sql_tool.py        # Text-to-SQL pipeline
│   │   └── rag_tool.py        # RAG retrieval + generation
│   ├── data/
│   │   ├── database.py        # DuckDB operations
│   │   └── vectorstore.py     # FAISS operations
│   ├── scoring/
│   │   └── quality.py         # Quality scoring module
│   └── ui/
│       ├── chat.py            # Chat rendering + visualization
│       └── sidebar.py         # Sidebar components
├── tests/
│   ├── test_sql_tool.py       # SQL tool tests (7 tests)
│   ├── test_rag_tool.py       # RAG tool tests (8 tests)
│   └── test_quality.py        # Quality scoring tests (10 tests)
└── data/
    ├── raw/                   # CSV + PDF source files
    └── processed/             # DuckDB + FAISS index files
```

## Testing

```bash
# Run all 32 tests
pytest tests/ -v
```

## Cost Estimate

Using OpenAI GPT-4o-mini + text-embedding-3-small:
- ~$0.001-0.002 per question
- ~$0.003 for one-time PDF embedding (184 chunks)
- ~$1 covers 500-1000 questions

## Known Limitations

- Dataset is simulated (not real fraud data)
- Single LLM provider (OpenAI) — no fallback
- FAISS index is in-memory (fine for ~184 chunks)
- No conversation memory across sessions (stateless per Streamlit session)
