# System Architecture & Design Document: AI-Powered Restaurant Operations Assistant

## 1. Executive Summary & Problem Framing
Restaurant multi-unit operators and general managers handle hundreds of customer reviews across platforms like Yelp. Manually parsing reviews to identify recurring operational bottlenecks (kitchen delays, cold food, allergen errors, rude staff, billing issues) is labor-intensive, reactive, and inconsistent.

This system provides an end-to-end AI-powered microservice and React interface that:
1. Performs **hybrid semantic and keyword retrieval** over the Yelp Open Dataset.
2. Coordinates an autonomous **multi-agent pipeline** that validates inputs, classifies complaints, prioritizes urgent issues (P0-P3 SLAs), diagnoses operational root causes via Aspect-Based Sentiment Analysis (ABSA), and drafts standard operating procedures (SOPs).
3. Evaluates recommendation quality using **DeepEval** and an **LLM-as-judge** operational audit.
4. Predicts **Customer Satisfaction (CSAT) and Net Promoter Score (NPS)** using trained machine learning models.
5. Models **complaint velocity and temporal trends** to forecast surges in friction categories.
6. Implements a **continuous learning feedback loop** persisting manager feedback in SQLite to adapt prompt exemplars and retrieval weights.

---

## 2. Architecture Overview & Data Flow
The architecture is structured across eight enterprise production layers:
```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 8: REACT 18 SPA DASHBOARD (Port 3000)                     │
│  [Operations Assistant]   [Review Explorer]   [Trends]   [CSAT Predictor]   [Eval Center]│
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ HTTP REST / JSON API
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                    LAYER 5: 6-TIER RESILIENT LLM CASCADE & ENGINE                       │
│  [1. Anthropic] ──> [2. NVIDIA] ──> [3. OpenRouter] ──> [4. Gemini] ──> [5. OpenAI]    │
│  └───────────────> [6. Dynamic Local NLP Fallback] (180s Circuit Breaker)               │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                 LAYER 3: MULTI-AGENT COLLABORATION PIPELINE (FastAPI 8000)             │
│  [1. Guardrails] ──> [2. Fast Semantic Router (0.44ms)] ──> [3. Hybrid Retrieval]       │
│  ──> [4. Taxonomy Classifier] ──> [5. Priority Engine (P0-P4)] ──> [6. ABSA Forensics] │
│  ──> [7. Operations Strategist] ──> [8. Executive Synthesis] ──> [9. LLM Judge]        │
└──────────────┬───────────────────────────┬───────────────────────────┬─────────────────┘
               │                           │                           │
┌──────────────▼─────────────┐ ┌───────────▼─────────────┐ ┌───────────▼─────────────────┐
│ LAYER 2: HYBRID RETRIEVAL  │ │ LAYER 4: QUALITY AUDIT  │ │ LAYER 6: ML & FORECASTING   │
│ • BGE-M3 (1024-dim, MPS)   │ │ • DeepEval Suite        │ │ • TF-IDF + Ridge CSAT Reg   │
│ • Persistent BM25Okapi     │ │   - Faithfulness        │ │ • NPS & Star Rating Predict │
│ • Reciprocal Rank Fusion   │ │   - Answer Relevancy    │ │ • Trend Velocity Forecaster │
│ • BGE Reranker v2 M3 Cross │ │ • LLM-as-Judge 4-Rubric │ │ • Next-Month Surge Alerter  │
│ • Evidence Affirmation Gate│ │ • Operational Pass Gate │ │ • Linguistic Driver Forensics│
└──────────────┬─────────────┘ └───────────┬─────────────┘ └───────────┬─────────────────┘
               │                           │                           │
┌──────────────▼───────────────────────────▼───────────────────────────▼─────────────────┐
│ LAYER 7: CONTINUOUS LEARNING FEEDBACK LOOP                                             │
│ • SQLite (`data/manager_feedback.db`) • Dynamic Few-Shot Exemplars • Adaptive Alpha    │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│ LAYER 1: ENTERPRISE DATA & VECTOR PERSISTENCE TIER                                     │
│ • Yelp Dataset (150k businesses, 1.5k reviews) • PostgreSQL Database                   │
│ • Qdrant Vector Engine (Port 6333)             • High-Speed In-Memory Review Cache     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core System Design Decisions & Trade-Offs

### Decision 1: Hybrid Retrieval (Dense Vector + BM25) vs. Pure Semantic Search
* **Trade-off Considered**: Using pure vector embeddings (e.g. OpenAI or SentenceTransformers) vs. pure BM25 keyword matching vs. hybrid fusion.
* **Analysis**:
  - Pure semantic search excels at finding conceptual matches (e.g., query "lukewarm soup" matches "broth had no steam and was cold"), but often misses critical domain-specific exact tokens (e.g., specific dish names like "chicken parmigiana", restaurant names like "Bella Roma", or specific error codes).
  - Pure BM25 handles exact keyword tokens cleanly, but fails on semantic paraphrasing and synonyms.
* **Decision**: Implemented **Hybrid Search** with Reciprocal Rank Fusion (RRF) and tunable $\alpha$-weighted score blending:
  $$\text{Score}_{hybrid} = \alpha \cdot \text{Score}_{semantic} + (1 - \alpha) \cdot \text{Score}_{BM25}$$
  Managers can dynamically slide $\alpha$ from `0.0` (BM25 keyword) to `1.0` (Dense vector) via the frontend UI.

### Decision 2: Centralized 6-Tier LLM Provider Cascade & Resilient Routing
* **Trade-off Considered**: Single cloud LLM provider vs. uncoordinated individual agent calls vs. centralized cascade router.
* **Analysis**:
  - Hardcoding a single provider creates single-point-of-failure risks during outages, rate limit throttles, or network connection timeouts.
  - Allowing each agent to implement its own LLM routing creates fragmented retry logic and duplicate API key parsing.
* **Decision**: Implemented a **Centralized 6-Tier LLM Cascade Engine** in `backend/app/llm/engine.py`:
  1. **Anthropic Claude (Primary)**: `claude-sonnet-5-5`
  2. **NVIDIA NIM (Secondary)**: `openai/gpt-oss-20b` (via `https://integrate.api.nvidia.com/v1`)
  3. **OpenRouter (Tertiary)**: `google/gemini-2.5-flash` (via `https://openrouter.ai/api/v1`)
  4. **Google Gemini (Fourth)**: `gemini-1.5-flash`
  5. **OpenAI (Fifth)**: `gpt-4o-mini`
  6. **Dynamic Local Reasoning Fallback (Sixth/Final)**: Fully offline dynamic NLP generator formulated on actual retrieved complaint entities and domain aspects.
  - **Resilience Controls**: Bounded retries (max 2) with exponential backoff on transient errors (HTTP 429, 502, 503, 504), `Retry-After` header parsing, immediate fail-fast on permanent authentication (401/403) or missing model (404) errors, configurable timeouts, explicit provider overrides for debugging, and secret-safe logging.

### Decision 3: Mathematical Multi-Factor Prioritization Formula vs. Qualitative Severity
* **Trade-off Considered**: Simple high/medium/low LLM classification vs. deterministic formulaic scoring.
* **Analysis**: Uncalibrated LLMs often over-escalate minor complaints or under-escalate quiet legal/hygiene risks.
* **Decision**: Implemented an explicit mathematical scoring algorithm:
  $$\text{Priority Score} = 0.35 \cdot \text{Severity} + 0.25 \cdot \text{Urgency} + 0.20 \cdot \text{RatingPenalty} + 0.20 \cdot \text{Frequency}$$
  - **Critical Override Guard**: If critical foodborne illness (e.g. poisoning, hospitalization) or severe allergen violations are detected, the system overrides scores to **P0 - Critical** (Score $\ge 92/100$) with a strict 1-hour SLA.
  - **Negation & Idiom Safeguards**: Sanitizes non-illness idioms (e.g., "never get sick of") to prevent false-positive P0 emergency escalations.

### Decision 4: Production Hybrid RAG & BGE Reranker Robustness
* **Trade-off Considered**: Pure vector vs. hybrid RRF with cross-encoder reranking.
* **Analysis**: Rerankers operating on multi-threaded web servers can trigger shape mismatch exceptions or race condition crashes if array dimensions and logit representations are unsanitized.
* **Decision**: Implemented `_safe_rerank` in `ProductionRetriever`:
  1. Thread-safe execution via `threading.Lock()` mutex.
  2. Universal normalization of 1D/2D/scalar numpy and tensor output formats into 1D Python float lists.
  3. Strict monotonic descending sorting by `reranker_score`.
  4. Clean fallback to Reciprocal Rank Fusion (`rrf_score`) on any model exception, ensuring zero search request failures.
  5. Strict Evidence Gate with idiom stripping ("sick of") and negation removal ("nobody got sick").

### Decision 5: DeepEval & LLM-as-Judge Dual Evaluation
* **Trade-off Considered**: Single evaluation metric vs. multi-perspective audit.
* **Analysis**: Recommendations must satisfy both textual factual consistency (DeepEval Faithfulness & Relevancy) and operational floor practicality (LLM-as-judge Actionability & Calibration).
* **Decision**: Embedded **DeepEval's standard metrics** alongside an **Executive Judge Agent** that audits every recommendation against a 4-point operational rubric before delivering it to the general manager.

---

## 4. Yelp Dataset Schema & Integration
The system integrates the standard Yelp Open Dataset format:
| Field | Type | Description |
|---|---|---|
| `business_id` | String | Unique restaurant business identifier |
| `name` | String | Commercial name of the restaurant |
| `categories` | String | Comma-delimited business attributes & cuisines |
| `city` | String | Location city (e.g. Philadelphia, Austin, etc.) |
| `state` | String | Location 2-letter state code |
| `stars` | Float | Aggregate business star rating (1.0 to 5.0) |
| `review_count` | Int | Cumulative historical review count |
| `review_id` | String | Unique review identifier |
| `user_id` | String | Review author identifier |
| `stars` | Int | Specific review rating |
| `text` | String | Full verbatim customer feedback text |
| `useful/funny/cool` | Int | Community vote signals |
| `date` | String | ISO timestamp (YYYY-MM-DD HH:MM:SS) |

---

## 5. Security & Validation Guardrails
The input pipeline enforces four distinct security and sanitization checks:
1. **Length Boundaries**: Queries must be between 3 and 1,200 characters.
2. **Prompt Injection Filtration**: Regex patterns detect and reject jailbreak attempts (`"ignore previous instructions"`, `"system prompt"`, `"DROP TABLE"`, etc.).
3. **Domain Relevance**: Queries are evaluated against hospitality terminology to warn users if the input diverges from restaurant operations.
4. **HTML/XSS Sanitization**: Control characters and script tags are stripped before processing.
