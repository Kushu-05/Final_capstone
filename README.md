<Question ID="1" Shortcut="chapter 1" Order="" ElementType="chapter" >
<LongCaption># AI-Powered Restaurant Operations & Review Assistant</LongCaption>
</Question>

An enterprise-grade semantic retrieval and multi-agent assistant built for restaurant general managers and multi-unit operators. Built on the **Yelp Open Dataset**, this system combines **Hybrid Search (Dense Semantic + BM25)**, **Multi-Agent Operations Collaboration**, **DeepEval Quality Evaluation**, **Customer Satisfaction (CSAT) Prediction**, **Complaint Trend Forecasting**, and an **SQLite Continuous Feedback Loop**.

---

## Architecture Overview

The system is built on an enterprise 8-layer architecture combining vector search, sparse keyword matching, neural cross-encoder reranking, multi-agent LLM orchestration, ML forecasting, and a closed-loop manager feedback system.

Refer to the visual architecture deliverables:
- **PNG (Lossless High-Resolution)**: [`architecture_diagram.png`](file:///Users/bharathyadav/Downloads/project6/architecture_diagram.png)
- **JPEG (Web-Optimized)**: [`architecture_diagram.jpeg`](file:///Users/bharathyadav/Downloads/project6/architecture_diagram.jpeg)
- **PDF (Vector Document)**: [`architecture_diagram.pdf`](file:///Users/bharathyadav/Downloads/project6/architecture_diagram.pdf)
- **System Design Document**: [`DESIGN.md`](file:///Users/bharathyadav/Downloads/project6/DESIGN.md)
- **Panel Presentation Deck**: [`PRESENTATION.md`](file:///Users/bharathyadav/Downloads/project6/PRESENTATION.md)

### The 8 Enterprise Architectural Layers

1. **Layer 1: Enterprise Data & Vector Persistence**
   - **Yelp Open Dataset**: 150,000+ businesses and 1,500+ curated restaurant review records.
   - **Qdrant Vector Engine** (Port 6333): High-performance vector database storing 1024-dimensional dense embeddings with HNSW indexing and payload filtering.
   - **Relational & In-Memory Storage**: PostgreSQL schemas and high-speed in-memory review caches for sub-millisecond document lookups.

2. **Layer 2: Production Hybrid Retrieval Engine**
   - **Dense Neural Search**: BAAI `bge-m3` (1024-dim) running with PyTorch MPS acceleration on Apple Silicon (or CPU).
   - **Sparse Lexical Search**: Persistent `BM25Okapi` inverted index for exact dish, allergen, and brand token matching.
   - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse ranks with standard constant $k=60$ alongside tunable score blending:
     $$\text{Score}_{hybrid} = \alpha \cdot \text{Score}_{semantic} + (1 - \alpha) \cdot \text{Score}_{BM25}$$
   - **BGE Reranker v2 M3 Cross-Encoder**: Deep cross-attention score re-ranking with thread-safe lock management.
   - **Evidence Affirmation Gate**: Strips non-illness idioms (e.g., "never get sick of") and filters negations ("nobody got sick") before analysis.

3. **Layer 3: Multi-Agent Operational Pipeline**
   - **Input Guardrails**: Character boundaries, prompt injection filters, and out-of-domain query detection.
   - **Fast Semantic Router**: 0.44ms zero-LLM embedding classification router with immediate bypass for direct lookups.
   - **Complaint Taxonomy Classifier**: Classifies friction points into 7 categories (Food Quality, Staff Service, Wait Time, Hygiene, Order Accuracy, Pricing, Ambience).
   - **Mathematical Prioritizer (P0–P4)**: Multi-factor severity algorithm ($0.35 \times \text{Severity} + 0.25 \times \text{Urgency} + 0.20 \times \text{RatingPenalty} + 0.20 \times \text{Frequency}$) with 1h SLA overrides for foodborne illness and allergen emergencies.
   - **Root-Cause Aspect Forensics (ABSA)**: Identifies operational failure points (e.g., kitchen pass heat lamp decay, host stand wave seating).
   - **Operations Strategist**: Prescribes immediate 24h floor actions, standard operating procedures (`SOP-OPS-202`, `SOP-HACCP-401`), and staff coaching plans.
   - **Executive Synthesis**: Condensed 1-paragraph brief for fast restaurant general manager consumption.
   - **LLM-as-Judge Auditor**: Four-rubric validation (Grounding, Actionability, Severity Calibration, Feasibility) requiring $\ge 80\%$ score to approve.

4. **Layer 4: Quality Audit & Evaluation Framework**
   - **DeepEval Test Suite**: Automated evaluation measuring Faithfulness, Answer Relevancy, Aspect Coverage, and Operational Feasibility.
   - **LLM-as-Judge 4-Rubric Operational Audit**: Automated pass/fail gate for operational safety.

5. **Layer 5: 6-Tier Resilient LLM Cascade & Router**
   - **Tier 1**: Anthropic Claude (`claude-sonnet-5-5`)
   - **Tier 2**: NVIDIA NIM (`openai/gpt-oss-20b`)
   - **Tier 3**: OpenRouter (`google/gemini-2.5-flash`)
   - **Tier 4**: Google Gemini (`gemini-1.5-flash`)
   - **Tier 5**: OpenAI (`gpt-4o-mini`)
   - **Tier 6**: Dynamic Local NLP Fallback (offline zero-hardcoding extraction)
   - **Circuit Breakers & Backoff**: Exponential backoff on 429/503 errors, 180s circuit breaker per provider, and instant fallback.

6. **Layer 6: Machine Learning Analytics & Forecasting**
   - **CSAT & NPS Regressor**: Scikit-Learn TF-IDF N-Gram + Ridge Regression predicting satisfaction % (0–100%), star rating (1–5), and NPS proxy.
   - **Trend Forecaster**: Time-series linear slope analysis estimating monthly complaint velocity and alerting on friction surges.

7. **Layer 7: Continuous Learning Feedback Loop**
   - **SQLite Persistence**: Stores manager ratings, corrective feedback, and thumbs-up/down verdicts in `data/manager_feedback.db`.
   - **Adaptive Exemplars**: Injects high-rated historical recommendations into few-shot multi-agent prompts.

8. **Layer 8: React 18 SPA Operations Dashboard**
   - 5 dedicated views: Restaurant Operations Assistant, Review Explorer, Trend Forecasting, CSAT Predictor, DeepEval Quality Evaluation.

---

## Restaurant Operations Assistant vs. Review Explorer

A key architectural distinction exists between the **Restaurant Operations Assistant** and the **Review Explorer**:

| Feature / Dimension | Restaurant Operations Assistant | Review Explorer |
|---|---|---|
| **Primary Audience** | General Managers, District Operators, Head Chefs | Business Analysts, Shift Supervisors, Data Auditors |
| **Core Purpose** | Proactive problem solving and decision formulation | Raw data discovery and semantic inspection |
| **Workflow** | Autonomous 9-step multi-agent collaboration pipeline | Direct, transparent hybrid retrieval querying |
| **Output Provided** | Priority tier (P0–P4), SLA timeline, ABSA root cause, immediate 24h floor steps, preventative SOPs, 1-paragraph GM brief, and LLM Judge verdict | Ranked list of verbatim Yelp customer reviews, similarity scores, dense/sparse contribution breakdown, and metadata |
| **Interactivity** | Issue prompts ("Customers complain of cold pasta..."), one-click thumbs up/down feedback to SQLite | Filter by restaurant, city, category, star rating, and fine-tune the $\alpha$ slider ($0.0 \leftrightarrow 1.0$) in real time |
| **LLM Involvement** | Full 6-tier LLM cascade with prompt engineering & reasoning | Pure local neural retrieval (BGE-M3 + BM25 + BGE Reranker) without LLM token cost |

---

## Key Capabilities & Requirements

| Module | Features & Capabilities |
|---|---|
| **Hybrid Search RAG** | Combines Dense Vector Embeddings (`BAAI/bge-m3` 1024-dim) + Sparse BM25 (`BM25Okapi`) + Qdrant Vector Engine with tunable $\alpha$ slider ($0.0 \leftrightarrow 1.0$) and Reciprocal Rank Fusion (RRF $k=60$). |
| **BGE Reranker v2 M3** | Cross-encoder reranker providing deep query-document relevance scoring with thread-safe lock management. |
| **Input Guardrails** | Validates length boundaries, prompt injection attempts, toxic inputs, and off-topic domain queries. |
| **Fast Semantic Router** | 0.44ms intent classifier routing standard complaints directly to hybrid retrieval without LLM latency. |
| **Complaint Classification** | Maps issues into operational taxonomies (Food Quality, Staff Service, Wait Time, Hygiene, Order Accuracy, Pricing, Ambience). |
| **Complaint Prioritization** | Mathematical severity formula prioritizing issues from P0 (Critical 1h SLA) down to P4 (Low), with automatic emergency overrides for health hazards and allergens. |
| **Aspect & Root Cause Analysis** | Aspect-Based Sentiment Analysis (ABSA) extracting friction points and identifying operational bottlenecks (FOH communication, BOH line queuing, prep temperature hold decay). |
| **Operations Strategist** | Formulates immediate 24h floor actions, preventative SOPs (`SOP-OPS-202`, `SOP-HACCP-401`), staff coaching plans, and measurable KPI benchmarks. |
| **Executive Synthesis** | Concise 1-paragraph summary distilling complex complaints into high-impact operational directives. |
| **LLM-as-Judge Evaluator** | Multi-rubric audit scoring Grounding, Precision, Actionability, and Severity Calibration (1-5 scale) with automated approval verdicts. |
| **DeepEval Suite** | Automated evaluation testing Faithfulness, Answer Relevancy, Aspect Coverage, and Operational Feasibility. |
| **CSAT Predictor** | Machine learning regressor (TF-IDF N-Gram + Ridge Regression) predicting customer satisfaction (0-100%), star ratings (1-5), and Net Promoter Score (NPS). |
| **Trend Forecaster** | Time-series regression modeling historical complaint velocity and projecting next-month risk surges. |
| **Continuous Feedback Loop** | SQLite persistence (`data/manager_feedback.db`) tracking manager ratings and adapting few-shot exemplars dynamically. |
| **6-Tier LLM Cascade Router** | Automated fallback: Anthropic Claude $\rightarrow$ NVIDIA NIM $\rightarrow$ OpenRouter $\rightarrow$ Gemini $\rightarrow$ OpenAI $\rightarrow$ Local NLP with 180s circuit breaker. |

---

## Project Structure

```
project6/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI microservice routes & CORS
│   │   ├── config.py                # Provider & API key configuration
│   │   ├── dataset/
│   │   │   └── yelp_dataset.py      # Yelp Open Dataset loader & generator
│   │   ├── retrieval/
│   │   │   ├── hybrid_search.py     # Dense vector + BM25Okapi hybrid search
│   │   │   └── production_retriever.py # BGE-M3 + Qdrant + BGE Reranker v2 M3
│   │   ├── agents/
│   │   │   ├── guardrails.py        # Input validation guardrails
│   │   │   ├── semantic_router.py   # 0.44ms fast embedding intent router
│   │   │   ├── classifier.py        # Complaint taxonomy classifier
│   │   │   ├── prioritizer.py       # Mathematical priority (P0-P4) & SLA engine
│   │   │   ├── root_cause_aspect.py # Aspect-Based Sentiment Analysis (ABSA) forensics
│   │   │   ├── operations_strategist.py # Immediate actions & preventative SOPs
│   │   │   ├── executive_summary.py # 1-paragraph GM executive synthesis
│   │   │   ├── llm_judge.py         # 4-rubric operational audit & pass/fail verdict
│   │   │   └── multi_agent_pipeline.py # End-to-end multi-agent pipeline orchestrator
│   │   ├── analytics/
│   │   │   ├── csat_predictor.py    # CSAT & NPS ML predictor (Ridge Regression)
│   │   │   └── trend_predictor.py   # Temporal trend & surge velocity forecasting
│   │   ├── evaluation/
│   │   │   └── deepeval_runner.py   # DeepEval quality evaluation suite
│   │   ├── feedback/
│   │   │   └── feedback_loop.py     # SQLite continuous learning feedback loop
│   │   └── llm/
│   │       └── engine.py            # 6-Tier unified LLM engine with circuit breakers
│   ├── tests/
│   │   └── test_api_endpoints.py    # Pytest automated API test suite (8/8 passing)
│   ├── bin/
│   │   └── qdrant                   # Qdrant Vector Engine binary
│   └── config/
│       └── qdrant.yaml              # Qdrant vector database configuration
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx           # Top navigation bar
│   │   │   ├── SettingsModal.jsx    # API key & 6-tier engine selector modal
│   │   │   ├── MultiAgentView.jsx   # Restaurant Operations Assistant
│   │   │   ├── SearchExplorerView.jsx # Review Explorer
│   │   │   ├── TrendPredictionView.jsx # Recharts trend charts & forecasting
│   │   │   ├── CSATPredictorView.jsx# ML CSAT & star predictor
│   │   │   └── DeepEvalFeedbackView.jsx # DeepEval & SQLite feedback loop
│   │   ├── api.js                   # REST API client
│   │   ├── App.jsx                  # Main React application
│   │   ├── index.css                # Tailwind CSS directives
│   │   └── main.jsx                 # React DOM mount point
│   ├── dist/                        # Production build artifacts
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── data/
│   ├── yelp_businesses.json         # Yelp business records (150k sample)
│   ├── yelp_reviews.json            # Yelp review records (1.5k curated)
│   └── manager_feedback.db          # SQLite feedback persistence database
├── docs/
│   └── generate_architecture_diagram.py # Script generating 300 DPI architecture visuals
├── start.sh                         # 1-Click launcher (Qdrant + FastAPI + Frontend)
├── stop.sh                          # 1-Click shutdown & port cleanup script
├── architecture_diagram.png         # High-resolution architecture visual (300 DPI)
├── architecture_diagram.jpeg        # Deliverable 1
├── architecture_diagram.pdf         # Deliverable 1
├── DESIGN.md                        # Deliverable 2: System Architecture & Design
├── PRESENTATION.md                  # Deliverable 4: Panel Presentation Deck
└── README.md                        # Deliverable 3: Project Documentation & Run Guide
```

---

## Quickstart & Steps to Run

Follow these instructions to run the application on your machine.

### 1. Prerequisites
- **Operating System**: macOS (tested on Apple Silicon M-series and Intel) or Linux / Windows (WSL2).
- **Python**: 3.10+ (Tested on Python 3.13)
- **Node.js**: v18+ (Tested on Node v22.18 / npm 11)
- **Network Ports**: Ensure ports `3000` (Frontend), `8000` (FastAPI), and `6333` (Qdrant) are available.

---

### 2. Method A: One-Click Startup (Recommended)

The easiest way to start all three services (Qdrant, Backend, Frontend) with health verification is using the automated control scripts:

```bash
# 1. Clone or navigate to the project root:
cd /Users/bharathyadav/Downloads/project6

# 2. Make scripts executable (if not already):
chmod +x start.sh stop.sh

# 3. Launch the full system:
./start.sh
```

#### What `./start.sh` does automatically:
1. Clears any lingering processes on ports `6333`, `8000`, and `3000`.
2. Starts the **Qdrant Vector Database** (`./backend/bin/qdrant`) on port `6333`.
3. Starts the **FastAPI Backend** with `uvicorn` on port `8000`.
4. Polls the backend health check (`http://127.0.0.1:8000/api/health`) until the BGE-M3 neural model is warm and ready.
5. Builds and serves the **React 18 Frontend** on port `3000`.

#### To stop all services:
```bash
./stop.sh
```
This cleanly terminates all background processes via their stored PIDs and clears ports `3000`, `8000`, and `6333`.

#### Viewing live logs:
```bash
tail -f logs/backend.log     # FastAPI backend logs & LLM cascade diagnostics
tail -f logs/qdrant.log      # Qdrant vector database logs
tail -f logs/frontend.log    # React frontend logs
```

---

### 3. Method B: Step-by-Step Manual Execution

If you prefer running each service in a separate terminal window:

#### Terminal 1 — Start Qdrant Vector Engine:
```bash
cd /Users/bharathyadav/Downloads/project6
./backend/bin/qdrant --config-path backend/config/qdrant.yaml
```
*Running on `http://127.0.0.1:6333`.*

#### Terminal 2 — Start FastAPI Backend:
```bash
cd /Users/bharathyadav/Downloads/project6
source venv/bin/activate
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Running on `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.*

#### Terminal 3 — Start React Frontend:
```bash
cd /Users/bharathyadav/Downloads/project6/frontend
npm install
npm run build
# Serve the production build on port 3000:
python3 -m http.server 3000 --directory dist
# OR run the Vite dev server:
# npm run dev
```
*Running on `http://127.0.0.1:3000`.*

---

### 4. Service URLs & Endpoints

| Service | URL | Description |
|---|---|---|
| **Web Dashboard** | [`http://127.0.0.1:3000`](http://127.0.0.1:3000) | Full React 18 user interface (all 5 tabs) |
| **Interactive API Docs** | [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) | Swagger UI for exploring and testing endpoints |
| **Backend Health Check** | [`http://127.0.0.1:8000/api/health`](http://127.0.0.1:8000/api/health) | Reports engine status, provider health, model readiness |
| **Qdrant Vector DB** | [`http://127.0.0.1:6333`](http://127.0.0.1:6333) | Vector database REST API and storage status |

---

### 5. Automated Verification & Testing

Run the full end-to-end API test suite to verify that all modules are working:

```bash
# In the project root:
./venv/bin/pytest -v backend/tests/test_api_endpoints.py
```
*Expected result: `8 passed` across health, hybrid search, classifier, prioritizer, CSAT predictor, trend forecaster, multi-agent pipeline, and feedback loop.*

#### Quick Curl Tests:
- **Test Hybrid Search**:
  ```bash
  curl -s -X POST http://127.0.0.1:8000/api/search \
    -H "Content-Type: application/json" \
    -d '{"query": "cold food and slow service", "limit": 3, "alpha": 0.5}' | jq .
  ```

- **Test Multi-Agent Operations Pipeline**:
  ```bash
  curl -s -X POST http://127.0.0.1:8000/api/multi-agent \
    -H "Content-Type: application/json" \
    -d '{"complaint": "Customers are repeatedly complaining about slow service and cold pasta at Bella Roma Trattoria."}' | jq .
  ```

---

## Dual-Engine & LLM Provider Cascade

The system implements a centralized, production-grade LLM routing mechanism with bounded retries, exponential backoff, secret-safe diagnostic logging, circuit breakers, and a strict 6-tier fallback priority:

1. **Anthropic Claude (Primary)** — `claude-sonnet-5-5`
2. **NVIDIA NIM (Secondary)** — `openai/gpt-oss-20b` (via `https://integrate.api.nvidia.com/v1`)
3. **OpenRouter (Tertiary)** — `google/gemini-2.5-flash` (via `https://openrouter.ai/api/v1`)
4. **Google Gemini (Fourth)** — `gemini-1.5-flash` (via `https://generativelanguage.googleapis.com/v1beta`)
5. **OpenAI (Fifth)** — `gpt-4o-mini`
6. **Dynamic Local Operations Engine (Final Fallback)** — Fully offline zero-hardcoded NLP domain analyzer generating structured root causes, SOPs, and KDS pacing actions.

   ```
2. **Via the UI Settings Modal**:
   - Click the **Gear icon** in the top-right corner of the React dashboard.
   - Configure keys and select your preferred active provider.
   - Click **Save & Apply**.

---

## Sample Usage & End-to-End Test Scenarios

### Test Scenario 1: Kitchen Ticket Pacing & Cold Food
- **Query**:
  ```text
  Customers are repeatedly complaining about slow service at Bella Roma Trattoria. What similar complaints have been reported before, and what issues are associated with them?
  ```
- **Execution Flow**:
  1. *Guardrail Agent* validates query safety.
  2. *Hybrid Retrieval* extracts grounded Yelp reviews for *Bella Roma Trattoria* (e.g. 55-minute ticket backlogs, cold chicken parmigiana).
  3. *Classifier* flags `Wait Time` and `Food Quality`.
  4. *Prioritizer* assigns `P1 - High` (Score ~78/100, 4-hour SLA).
  5. *Aspect & Root-Cause Agent* diagnoses pass-window thermal decay and wave-seating ticket stacking.
  6. *Operations Strategist* recommends:
     - 90-second runner pickup rule.
     - `SOP-OPS-202`: KDS ticket pacing benchmarks.
     - 10-table host stand seating wave limits.
  7. *LLM-as-Judge* verifies evidence grounding and returns **APPROVED** (> 88%).

### Test Scenario 2: Emergency Food Safety & Contamination
- **Query**:
  ```text
  URGENT: Customer found a metal staple in their burger bun and raw seafood reports. What is the immediate safety protocol?
  ```
- **Result**:
  - Hazard override triggers **P0 - Critical** (Score 95/100).
  - SLA: Immediate action (< 1 hour).
  - Generates quarantine directives, `SOP-HACCP-401` sanitization protocols, and ServSafe re-certification mandates.

---

## API Microservice Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | `GET` | Health check and active engine status |
| `/api/config` | `GET`, `POST` | Get or update LLM provider and API keys |
| `/api/metadata` | `GET` | Distinct restaurants, cities, categories, review counts |
| `/api/search` | `POST` | Hybrid BM25 + dense semantic vector search |
| `/api/classify` | `POST` | Complaint category, sentiment, and urgency classification |
| `/api/prioritize` | `POST` | Priority score, tier (P0-P3), and SLA calculation |
| `/api/csat-predict` | `POST` | ML CSAT %, star rating, and NPS proxy prediction |
| `/api/trends` | `POST` | Temporal regression trajectory and next-month forecast |
| `/api/multi-agent` | `POST` | Full collaborative multi-agent operations pipeline |
| `/api/evaluate` | `POST` | DeepEval recommendation quality assessment suite |
| `/api/feedback` | `GET`, `POST` | Continuous learning manager feedback in SQLite |
