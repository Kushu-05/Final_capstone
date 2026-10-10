# Panel Presentation Guide (10 Minutes)
## AI-Powered Restaurant Operations & Review Assistant

**Time Allocation**:
- **8 Minutes**: Live Solution Demo & System Design Walkthrough
- **2 Minutes**: Technical Q&A & Panel Discussion

---

## 8-Minute Demo Walkthrough Plan

### Minute 0:00 - 1:00 • Problem Statement & Core Value Proposition
* **Key Talking Points**:
  - Restaurant businesses handle thousands of customer reviews across varied operational domains (food temperature, service pacing, staff demeanor, cleanliness, pricing, and allergen safety).
  - Manually searching reviews to identify systemic root causes is slow, reactive, and subjective.
  - Present **GastroOps**: An AI-powered semantic retrieval and multi-agent assistant built for multi-unit restaurant managers.
  - Mention key requirements: Hybrid search (BM25 + Semantic), multi-agent collaboration, DeepEval quality evaluation, CSAT prediction, trend forecasting, and continuous feedback loop.

### Minute 1:00 - 2:30 • Architecture & Dual-Engine Strategy
* **Show Deliverable**: Refer to [`architecture_diagram.png`](file:///Users/bharathyadav/Downloads/project6/architecture_diagram.png), [`architecture_diagram.jpeg`](file:///Users/bharathyadav/Downloads/project6/architecture_diagram.jpeg), and [`architecture_diagram.pdf`](file:///Users/bharathyadav/Downloads/project6/architecture_diagram.pdf).
* **Key Talking Points**:
  - **Zero Hardcoding Architecture**: Every response is dynamically computed via real embeddings, BM25, and statistical/neural synthesis.
  - **Dual-Engine LLM Router**: Pre-downloaded local embeddings (`all-MiniLM-L6-v2`) and dynamic local operations reasoning engine running offline.
  - **Seamless Cloud LLM Integration**: Can plug in OpenAI (`gpt-4o-mini`) or Google Gemini (`gemini-1.5-flash`) at runtime via `.env` or UI Settings modal.

### Minute 2:30 - 4:30 • Live Demo: Multi-Agent Copilot
* **Live Action**: Navigate to **Multi-Agent Copilot** tab in the React interface.
* **Test Scenario 1 (Kitchen Pacing & Cold Food)**:
  - Select quick prompt: *"Customers are repeatedly complaining about slow service at Bella Roma Trattoria. What similar complaints have been reported before?"*
  - Demonstrate the **Hybrid Search Alpha Slider** ($\alpha = 0.5$): Show how BM25 token matching and dense vector similarity are fused using Reciprocal Rank Fusion (RRF).
  - Click **Analyze & Solve**.
  - Show the **Multi-Agent Collaboration Trace**:
    1. Input Validation Guardrail Agent (passes injection & domain checks).
    2. Hybrid Retrieval Agent (retrieves grounded Yelp reviews).
    3. Classification & Prioritization Agent (computes priority score and SLA).
    4. Aspect & Root Cause Specialist Agent (diagnoses pass-window hold delays).
    5. Operations Strategist Agent (generates immediate 24h floor actions and preventative SOPs).
    6. Executive Quality Judge Agent (audits against a 4-criteria rubric).
* **Test Scenario 2 (Critical Food Safety Override)**:
  - Test query: *"URGENT: Customer found a metal staple in their burger bun and raw seafood reports."*
  - Highlight how the system immediately flags **P0 - Critical** (Score $\ge 92/100$) with immediate 1-hour SLA.

### Minute 4:30 - 5:45 • Live Demo: Trend Prediction & Category Risk
* **Live Action**: Switch to **Trend Predictor** tab.
* **Key Talking Points**:
  - Show the temporal time-series charts generated across historical Yelp review timestamps (2024–2026).
  - Point out the **Linear Regression Trajectory Slope ($m$)**: e.g., "SURGING" vs "STABLE".
  - Show the next-month forecasted complaint volume and the **Domain Distribution Bar Chart** (e.g. Wait Time vs Food Quality vs Cleanliness).

### Minute 5:45 - 6:45 • Live Demo: CSAT & Star Rating Predictor
* **Live Action**: Switch to **CSAT Predictor** tab.
* **Key Talking Points**:
  - Show how the Scikit-Learn TF-IDF N-Gram + Ridge Regression model dynamically predicts star ratings ($1.0 - 5.0$) and CSAT index ($0 - 100\%$).
  - Show the **Linguistic Driver Attribution**: Negative drag (e.g. "Food Thermal Decay -1.2 stars") vs. Positive lift drivers.

### Minute 6:45 - 8:00 • Live Demo: DeepEval Quality Audit & Continuous Feedback Loop
* **Live Action**: Switch to **DeepEval & Feedback** tab.
* **Key Talking Points**:
  - **DeepEval Suite**: Verify test results for Faithfulness, Answer Relevancy, Aspect Coverage, and Operational Feasibility.
  - **Continuous Feedback Loop**: Show how general managers submit ratings and custom notes, persisted in the SQLite database (`manager_feedback.db`).
  - Demonstrate how positive manager ratings adaptively inject few-shot exemplars and tune retrieval confidence!

---

## 2-Minute Q&A & Panel Preparation

### Anticipated Question 1:
> *"How do you handle latency during hybrid retrieval when searching thousands of reviews?"*
* **Response**:
  - Dense embeddings are pre-computed and L2-normalized upon startup. Cosine similarity reduces to an ultra-fast matrix-vector dot product in NumPy ($< 5\text{ ms}$).
  - BM25Okapi inverted index is tokenized in memory.
  - Metadata filtering is applied as a Boolean bitmask prior to fusion, minimizing scoring overhead.

### Anticipated Question 2:
> *"What prevents the model from hallucinating non-existent restaurant procedures?"*
* **Response**:
  - The pipeline employs a two-tier evaluation guard:
    1. **DeepEval Faithfulness Metric**: Checks lexical and semantic overlap between the recommended SOPs and the retrieved Yelp review context chunks.
    2. **Executive LLM-as-Judge Agent**: Audits evidence grounding (scored 1-5) and flags ungrounded assertions before executive sign-off.

### Anticipated Question 3:
> *"Can this system be connected to a live multi-restaurant POS or review webhook?"*
* **Response**:
  - Yes. The backend is built as a modular FastAPI microservice. The `YelpDatasetManager` supports streaming or batch JSON ingestion (`POST /api/search` and `POST /api/feedback`), allowing live webhook integration with Toast POS, Yelp Fusion API, or Google Business Reviews.
