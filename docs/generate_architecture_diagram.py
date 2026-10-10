import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_diagram():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    fig, ax = plt.subplots(figsize=(18, 12), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    c_primary = '#10B981'     # Emerald
    c_secondary = '#0EA5E9'   # Sky
    c_purple = '#8B5CF6'      # Violet
    c_amber = '#F59E0B'       # Amber
    c_rose = '#F43F5E'        # Rose
    c_card_bg = '#111827'     # Gray-900
    c_border = '#1F2937'      # Gray-800
    c_text = '#F9FAFB'        # White/Slate-50
    c_subtext = '#9CA3AF'     # Gray-400

    def draw_box(x, y, w, h, title, subtitle, items, border_color, accent_color):
        # Card background
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.8,rounding_size=1.2",
            facecolor=c_card_bg,
            edgecolor=border_color,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(rect)

        # Header tag
        tag = patches.FancyBboxPatch(
            (x + 1, y + h - 4), w - 2, 3.2,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor='#1E293B',
            edgecolor='none',
            zorder=3
        )
        ax.add_patch(tag)

        # Title
        ax.text(x + w/2, y + h - 2.5, title, color=accent_color, fontsize=11, fontweight='bold', ha='center', va='center', zorder=4)
        if subtitle:
            ax.text(x + w/2, y + h - 5.5, subtitle, color=c_subtext, fontsize=8, style='italic', ha='center', va='center', zorder=4)

        # Content items
        line_y = y + h - 8.5
        for item in items:
            ax.text(x + 2.5, line_y, f"• {item}", color=c_text, fontsize=8.5, ha='left', va='center', zorder=4)
            line_y -= 2.6

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color='#38BDF8', lw=2.2, mutation_scale=15),
            zorder=5
        )
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2 + 1.2
            ax.text(mx, my, label, color='#38BDF8', fontsize=7.5, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#0F172A', edgecolor='#334155', lw=1), zorder=6)

    # 1. Main Title
    ax.text(50, 97, "AI-POWERED RESTAURANT OPERATIONS & REVIEW ASSISTANT", 
            color='#FFFFFF', fontsize=18, fontweight='black', ha='center', va='center')
    ax.text(50, 94.2, "End-to-End Microservice Architecture • Hybrid RAG • Multi-Agent Collaboration • DeepEval • Feedback Loop", 
            color=c_primary, fontsize=9.5, fontweight='bold', ha='center', va='center')

    # ROW 1 (Top Left to Right): Dataset -> Hybrid Retrieval -> Multi-Agent Pipeline
    # 1. Dataset Layer
    draw_box(
        x=2, y=66, w=22, h=24,
        title="1. YELP OPEN DATASET & STORAGE",
        subtitle="Corpus Provenance & Databases",
        items=[
            "Authentic Yelp Corpus (150k biz, 1.5k reviews)",
            "PostgreSQL (Relational schema, B-tree indexes)",
            "Qdrant Vector DB (Port 6333, BGE-M3 collection)",
            "Persistent Sparse Index (bm25_index.pkl)",
            "In-Memory Review Cache (<0.1ms hydration)"
        ],
        border_color='#374151', accent_color=c_secondary
    )

    # 2. Ingestion & Retrieval Layer
    draw_box(
        x=27, y=66, w=24, h=24,
        title="2. PRODUCTION HYBRID RETRIEVAL",
        subtitle="Dense BGE-M3 + BM25 + Cross-Encoder",
        items=[
            "Dense Vector: BAAI/bge-m3 (1024-dim, MPS)",
            "Lexical Sparse: BM25Okapi Token Index",
            "Reciprocal Rank Fusion (RRF, k=60)",
            "BGE Reranker v2 M3 (Cross-Encoder, MPS)",
            "Evidence Affirmation Gate (No Hallucination)"
        ],
        border_color=c_secondary, accent_color=c_secondary
    )

    # 3. Multi-Agent Pipeline Layer
    draw_box(
        x=54, y=66, w=44, h=24,
        title="3. MULTI-AGENT COLLABORATION PIPELINE",
        subtitle="Specialized Autonomous Operations Agents",
        items=[
            "Agent 1: Input Guardrail Agent (Injection & domain checks)",
            "Agent 2: Fast Semantic Intent Router (0.44ms classification)",
            "Agent 3: Hybrid Retrieval & Filter Agent (RRF + Evidence Gate)",
            "Agent 4: Complaint Classifier (Taxonomy, Sentiment, Urgency)",
            "Agent 5: Mathematical Priority Engine (P0-P4 SLAs, Risk math)",
            "Agent 6: Aspect & Root-Cause Forensic Agent (FOH/BOH friction)",
            "Agent 7: Operations Strategist (Immediate actions & SOPs)",
            "Agent 8: Executive Review Synthesis Agent (1-paragraph answer)",
            "Agent 9: Executive Auditor & LLM Judge (Rubric quality audit)"
        ],
        border_color=c_primary, accent_color=c_primary
    )

    # Connecting Arrows Row 1
    draw_arrow(24, 78, 27, 78, "Vectors & Texts")
    draw_arrow(51, 78, 54, 78, "Ranked Evidence")

    # ROW 2 (Middle): Evaluation, 6-Tier Cascade, Analytics
    # 4. Advanced Evaluation (DeepEval & LLM Judge)
    draw_box(
        x=2, y=36, w=30, h=24,
        title="4. QUALITY AUDIT & EVALUATION",
        subtitle="LLM-as-Judge & DeepEval Suite",
        items=[
            "Evidence Grounding Rubric (Strict factual check)",
            "Root Cause Precision (FOH/BOH friction)",
            "Actionability & SOP Quality (Operational utility)",
            "Severity Calibration (P1-P4 SLA alignment)",
            "DeepEval Metric Suite (Faithfulness & Relevancy)",
            "Automated Verdict: APPROVED / NEEDS REVISION"
        ],
        border_color=c_rose, accent_color=c_rose
    )

    # 5. Dual LLM Provider Engine
    draw_box(
        x=35, y=36, w=30, h=24,
        title="5. 6-TIER LLM CASCADE ROUTER",
        subtitle="Centralized Fallback & Circuit Breaker",
        items=[
            "Tier 1: Anthropic Claude (claude-sonnet-5-5)",
            "Tier 2: NVIDIA NIM (meta/llama-3.2-11b)",
            "Tier 3: OpenRouter (google/gemini-2.5-flash)",
            "Tier 4: Google Gemini (gemini-1.5-flash)",
            "Tier 5: OpenAI API (gpt-4o-mini)",
            "Tier 6: Dynamic Local NLP Rules (Offline)",
            "Fail-Fast Circuit Breaker (180s Cooldown)"
        ],
        border_color=c_purple, accent_color=c_purple
    )

    # 6. Analytics & Trend Forecasting
    draw_box(
        x=68, y=36, w=30, h=24,
        title="6. CSAT & TREND PREDICTOR",
        subtitle="Machine Learning & Temporal Analytics",
        items=[
            "CSAT Regressor (TF-IDF + Ridge Regression)",
            "NPS Proxy Scoring & Star Rating Prediction",
            "Linguistic Driver Attribution (Positive vs Drag)",
            "Time-Series Complaint Velocity Trajectory",
            "Linear Trend Slope (m) & Surge Alerts",
            "Next 30-Day Operational Risk Forecasting"
        ],
        border_color=c_amber, accent_color=c_amber
    )

    # Arrows connecting Row 1 to Row 2
    draw_arrow(76, 66, 76, 60, "SOP Telemetry")
    draw_arrow(50, 48, 32, 48, "Evaluation Input")
    draw_arrow(65, 48, 68, 48, "Features & Reviews")

    # ROW 3 (Bottom): Feedback Loop & React Frontend
    # 7. SQLite Feedback Loop
    draw_box(
        x=2, y=6, w=30, h=24,
        title="7. CONTINUOUS FEEDBACK LOOP",
        subtitle="Human-in-the-Loop SQLite Persistence",
        items=[
            "Manager Feedback Logging (Thumbs up/down)",
            "1-5 Star Manager Satisfaction Ratings",
            "Accepted / Rejected Action Items & SOPs",
            "Manual Priority Tier Overrides & Shift Notes",
            "Approval Rate & Confidence Metrics",
            "Persistent Storage: data/manager_feedback.db"
        ],
        border_color=c_primary, accent_color=c_primary
    )

    # 8. React Interactive Frontend
    draw_box(
        x=35, y=6, w=63, h=24,
        title="8. REACT 18 DASHBOARD (Vite + TailwindCSS)",
        subtitle="Unified Web Interface on Port 3000",
        items=[
            "Tab 1: Operations Assistant (8-agent collaborative copilot, SOP cards, feedback)",
            "Tab 2: Review Explorer (Hybrid search, alpha slider, filters, executive summary)",
            "Tab 3: Complaint Trends (Temporal velocity, monthly volume charts, category bars)",
            "Tab 4: CSAT & NPS Predictor (Interactive review tester, sentiment driver attribution)",
            "Tab 5: DeepEval & Feedback Center (Quality assessment matrix, live SQLite history)",
            "Runtime Settings Modal: Provider selector (Claude, NVIDIA, Local) & API key config"
        ],
        border_color=c_secondary, accent_color=c_secondary
    )

    # Arrows for Row 3
    draw_arrow(50, 36, 50, 30, "FastAPI REST API (:8000)")
    draw_arrow(32, 18, 35, 18, "Feedback Ingestion")
    draw_arrow(66, 30, 66, 36, "Telemetry")

    plt.tight_layout()
    
    # Save PNG, JPEG, and PDF
    png_path = os.path.join(out_dir, "architecture_diagram.png")
    jpeg_path = os.path.join(out_dir, "architecture_diagram.jpeg")
    pdf_path = os.path.join(out_dir, "architecture_diagram.pdf")

    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(jpeg_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(pdf_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    
    print(f"Generated architecture diagrams successfully:\n - {png_path}\n - {jpeg_path}\n - {pdf_path}")

if __name__ == "__main__":
    generate_diagram()
