import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.config import config
from backend.app.dataset.yelp_dataset import dataset_manager
from backend.app.retrieval.hybrid_search import hybrid_retriever
from backend.app.retrieval.production_retriever import production_retriever
from backend.app.agents.guardrails import guardrails
from backend.app.agents.classifier import classifier_agent
from backend.app.agents.prioritizer import prioritizer_agent
from backend.app.agents.root_cause_aspect import aspect_root_cause_agent
from backend.app.agents.operations_strategist import operations_strategist_agent
from backend.app.agents.llm_judge import llm_judge_agent
from backend.app.agents.multi_agent_pipeline import multi_agent_pipeline
from backend.app.analytics.csat_predictor import csat_predictor
from backend.app.analytics.trend_predictor import trend_predictor
from backend.app.evaluation.deepeval_runner import deepeval_runner
from backend.app.feedback.feedback_loop import feedback_manager
from backend.app.agents.executive_summary import executive_summary_agent

app = FastAPI(
    title="AI-Powered Restaurant Operations & Review Assistant",
    description="Microservice for semantic retrieval, multi-agent operations, hybrid search, complaint prioritization, CSAT prediction, DeepEval evaluation, and feedback loops.",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Request Models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5
    business_id: Optional[str] = None
    category: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    min_stars: Optional[int] = None
    max_stars: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ClassifyRequest(BaseModel):
    text: str

class PrioritizeRequest(BaseModel):
    text: str
    stars: int = 1
    urgency: str = "High"
    category: str = "Food Quality"
    frequency_count: int = 1

class CSATRequest(BaseModel):
    text: str

class TrendRequest(BaseModel):
    business_id: Optional[str] = None
    category: Optional[str] = None

class MultiAgentRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5
    business_id: Optional[str] = None
    category: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    min_stars: Optional[int] = None
    max_stars: Optional[int] = None

class EvaluateRequest(BaseModel):
    query: str
    business_id: Optional[str] = None
    alpha: float = 0.5

class FeedbackRequest(BaseModel):
    query: str
    category: str = "General"
    is_helpful: bool = True
    rating: int = 5
    revised_priority: Optional[str] = None
    manager_notes: Optional[str] = ""
    accepted_actions: Optional[List[str]] = None

class ConfigUpdateRequest(BaseModel):
    provider: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

# API Endpoints
@app.get("/api/health")
def health_check():
    meta = dataset_manager.get_distinct_metadata()
    return {
        "status": "healthy",
        "active_llm_provider": config.get_active_provider(),
        "total_businesses": meta.get("total_restaurants", len(dataset_manager.businesses)),
        "total_reviews": meta.get("total_reviews", len(dataset_manager.reviews)),
        "retrieval_architecture": "BAAI/bge-m3 (1024-dim) + Qdrant + BM25 + RRF + BGE Reranker v2 M3",
        "local_embedding_model": "BAAI/bge-m3 (1024-dim, Loaded & Running Locally)"
    }

@app.get("/api/config")
def get_config():
    return {
        "active_provider": config.get_active_provider(),
        "anthropic_configured": bool(config.anthropic_api_key),
        "anthropic_model": config.anthropic_model,
        "nvidia_configured": bool(config.nvidia_api_key),
        "openrouter_configured": bool(config.openrouter_api_key),
        "openai_configured": bool(config.openai_api_key),
        "gemini_configured": bool(config.gemini_api_key),
        "nvidia_model": config.nvidia_model,
        "openrouter_model": config.openrouter_model,
        "openai_model": config.openai_model,
        "gemini_model": config.gemini_model,
        "local_fallback_available": True
    }

@app.post("/api/config")
def update_config(req: ConfigUpdateRequest):
    if req.anthropic_api_key:
        config.set_api_key("anthropic", req.anthropic_api_key)
    if req.nvidia_api_key:
        config.set_api_key("nvidia", req.nvidia_api_key)
    if req.openrouter_api_key:
        config.set_api_key("openrouter", req.openrouter_api_key)
    if req.openai_api_key:
        config.set_api_key("openai", req.openai_api_key)
    if req.gemini_api_key:
        config.set_api_key("gemini", req.gemini_api_key)
    if req.provider:
        config.preferred_provider = req.provider.lower()
    return {
        "message": "Configuration updated successfully",
        "active_provider": config.get_active_provider()
    }

@app.get("/api/metadata")
def get_metadata():
    return dataset_manager.get_distinct_metadata()

@app.get("/api/businesses")
def get_businesses():
    return dataset_manager.get_all_businesses()

@app.post("/api/search")
def hybrid_search(req: SearchRequest):
    is_valid, sanitized_q, guard_report = guardrails.validate_and_sanitize(req.query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=guard_report["flags"][0])

    import time, logging
    t_start = time.time()
    t0 = time.time()
    results = hybrid_retriever.hybrid_search(
        query=sanitized_q,
        top_k=req.top_k,
        alpha=req.alpha,
        business_id=req.business_id,
        category=req.category,
        city=req.city,
        state=req.state,
        min_stars=req.min_stars,
        max_stars=req.max_stars,
        start_date=req.start_date,
        end_date=req.end_date
    )
    t_ret = time.time() - t0

    diag = getattr(hybrid_retriever, "last_diagnostics", {})
    target_name = (
        diag.get("target_restaurant") or
        (req.business_id if (req.business_id and req.business_id.lower() != "all") else "All Restaurants")
    )
    t0 = time.time()
    summary_data = executive_summary_agent.generate(
        query=sanitized_q,
        reviews=results,
        target_restaurant=target_name,
        aspect=None
    )
    t_sum = time.time() - t0
    print(f"[/api/search Latency Profile] Total={time.time()-t_start:.3f}s (Retrieval={t_ret:.3f}s, Summary={t_sum:.3f}s)", flush=True)
    return {
        "query": sanitized_q,
        "total_retrieved": len(results),
        "alpha": req.alpha,
        "results": results,
        "executive_summary": summary_data.get("summary"),
        "key_reasons": summary_data.get("reasons", []),
        "diagnostics": diag
    }

@app.post("/api/classify")
def classify_complaint(req: ClassifyRequest):
    return classifier_agent.classify(req.text)

@app.post("/api/prioritize")
def prioritize_complaint(req: PrioritizeRequest):
    complaint_data = {
        "text": req.text,
        "stars": req.stars,
        "urgency": req.urgency,
        "category": req.category
    }
    return prioritizer_agent.prioritize(complaint_data, frequency_count=req.frequency_count)

@app.post("/api/csat-predict")
def predict_csat(req: CSATRequest):
    return csat_predictor.predict(req.text)

@app.post("/api/trends")
def predict_trends(req: TrendRequest):
    return trend_predictor.analyze_and_forecast(
        business_id=req.business_id,
        category=req.category
    )

@app.post("/api/multi-agent")
def run_multi_agent_pipeline(req: MultiAgentRequest):
    res = multi_agent_pipeline.run_collaboration(
        query=req.query,
        top_k=req.top_k,
        alpha=req.alpha,
        business_id=req.business_id,
        category=req.category,
        city=req.city,
        state=req.state,
        min_stars=req.min_stars,
        max_stars=req.max_stars
    )
    if not res.get("success", False):
        raise HTTPException(status_code=400, detail=res.get("error", "Multi-agent execution halted"))
    return res

@app.post("/api/evaluate")
def evaluate_recommendations(req: EvaluateRequest):
    # Runs full pipeline and DeepEval suite
    res = multi_agent_pipeline.run_collaboration(
        query=req.query,
        business_id=req.business_id,
        alpha=req.alpha,
        top_k=5
    )
    if not res.get("success", False):
        raise HTTPException(status_code=400, detail=res.get("error", "Evaluation failed"))

    deepeval_result = deepeval_runner.evaluate_recommendations(
        query=req.query,
        retrieved_reviews=res.get("retrieved_reviews", []),
        recommendations=res.get("recommendations", {}),
        root_cause_data=res.get("root_cause_analysis", {})
    )
    return {
        "query": req.query,
        "deepeval_suite": deepeval_result,
        "llm_judge_audit": res.get("judge_audit", {}),
        "recommendations": res.get("recommendations", {}),
        "retrieved_count": len(res.get("retrieved_reviews", []))
    }

@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    return feedback_manager.submit_feedback(
        query=req.query,
        category=req.category,
        is_helpful=req.is_helpful,
        rating=req.rating,
        revised_priority=req.revised_priority,
        manager_notes=req.manager_notes,
        accepted_actions=req.accepted_actions
    )

@app.get("/api/feedback")
def get_feedback():
    items = feedback_manager.get_all_feedback()
    metrics = feedback_manager.get_continuous_improvement_metrics()
    return {
        "metrics": metrics,
        "recent_feedback": items
    }
