import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import json
import numpy as np
from typing import Dict, Any, Optional, List
from sentence_transformers import SentenceTransformer

from backend.app.config import config
from backend.app.llm.engine import llm_engine

# Multi-exemplar prototypes per operational aspect for semantic embedding comparison
ASPECT_EXEMPLARS = {
    "Wait Time": [
        "Complaints about long waiting times, slow service, delayed orders.",
        "Service is late, orders take too long, waiting forever for a table or check.",
        "Why is service so slow, delays in food delivery."
    ],
    "Food Quality": [
        "Complaints about bad food quality, poor food taste, bad food.",
        "The food quality is bad, food is cold, undercooked, burnt, or unpalatable.",
        "Customers are unhappy with the food, food tastes terrible, bad meals."
    ],
    "Staff Behavior": [
        "Complaints about staff behavior, rude employees, bad customer service.",
        "The staff are rude, disrespectful waitstaff, inattentive servers.",
        "Customers don't like how employees behave, unprofessional staff demeanor."
    ],
    "Pricing": [
        "Complaints about pricing, expensive food, high prices.",
        "Customers think it is too expensive, the prices are too high.",
        "Overpriced menu items, high bill costs, unfair charges and fees."
    ],
    "Food Safety / Illness": [
        "Customers reporting getting sick after eating, food poisoning.",
        "Anyone become ill after eating, reports of food poisoning, nausea.",
        "Food safety issues, stomach illness, vomiting from food."
    ]
}

# Rich search representations tailored for BGE-M3 dense and BM25 sparse review retrieval
SEARCH_REPRESENTATIONS = {
    "Wait Time": "customer complaints about long waiting times, slow service, delayed orders, delays in receiving food",
    "Food Quality": "customer complaints about bad food quality, poor taste, cold undercooked dishes, unpalatable meals",
    "Staff Behavior": "customer complaints about rude staff behavior, bad service attitude, disrespectful waitstaff",
    "Pricing": "customer complaints about expensive prices, overpriced food, high bill costs, unfair charges",
    "Food Safety / Illness": "customer reports of getting sick after eating, food poisoning, nausea, becoming ill"
}

class QueryUnderstandingEngine:
    """
    Semantic Query Understanding Engine:
    Analyzes natural-language user queries BEFORE retrieval to extract:
    1. User Intent (complaint_analysis, root_cause, trend_analysis, severity_prioritization, recommendations, simple_retrieval)
    2. Primary Operational Aspect (Wait Time, Food Quality, Staff Behavior, Pricing, Food Safety / Illness, etc.)
    3. Semantic Search Representation (retrieval_query) for dual BGE-M3 Dense + BM25 Lexical retrieval.
    Preserves original_query and retrieval_query as separate state representations.
    """
    def __init__(self, embedder: Optional[SentenceTransformer] = None):
        self.embedder = embedder
        self.aspect_names = list(ASPECT_EXEMPLARS.keys())
        self.aspect_vectors: Dict[str, np.ndarray] = {}
        self._init_aspect_vectors()

    def _init_aspect_vectors(self):
        try:
            if self.embedder is None:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                from backend.app.retrieval.production_retriever import production_retriever
                if production_retriever.embedder is not None:
                    self.embedder = production_retriever.embedder
                else:
                    self.embedder = SentenceTransformer("BAAI/bge-m3")

            for name, texts in ASPECT_EXEMPLARS.items():
                vecs = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                self.aspect_vectors[name] = vecs
        except Exception as e:
            print(f"Warning: Could not pre-compute aspect vectors: {e}")
            self.aspect_vectors = {}

    def _detect_intent(self, query: str) -> str:
        """Determines operational intention from natural-language inquiry structure."""
        lq = query.lower().strip()
        if any(lq.startswith(w) or f" {w} " in lq for w in ["why", "what causes", "what caused", "root cause", "reason for", "reasons behind"]):
            return "root_cause"
        if any(lq.startswith(w) or f" {w} " in lq for w in ["how to fix", "how can we solve", "what should we do", "recommend", "recommendation", "action plan", "solution"]):
            return "recommendations"
        if any(w in lq for w in ["trend", "over time", "historically", "increase over", "spiking", "frequency trend"]):
            return "trend_analysis"
        if any(w in lq for w in ["how urgent", "priority", "severity level", "risk priority", "sla"]):
            return "severity_prioritization"
        if any(lq.startswith(w) for w in ["find review", "find reviews", "show review", "show reviews", "search review", "search reviews"]):
            return "simple_retrieval"
        return "complaint_analysis"

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Processes the raw user query into structured semantic metadata:
        - original_query: preserved verbatim
        - intent: complaint_analysis | root_cause | trend_analysis | severity_prioritization | recommendations | simple_retrieval
        - aspect: primary operational aspect (or None if ambiguous)
        - retrieval_query: semantic search representation
        """
        raw_query = query.strip()
        if not raw_query:
            return {
                "original_query": "",
                "retrieval_query": "",
                "intent": "complaint_analysis",
                "aspect": None,
                "confidence": 0.0,
                "method": "empty"
            }

        # 1. Fast Semantic Router via local BGE-M3 vector space representation (instantaneous)
        semantic_res = self._analyze_via_semantic_router(raw_query)
        if semantic_res.get("aspect") is not None or semantic_res.get("confidence", 0) >= 0.60:
            return semantic_res

        # 2. Fall back to LLM Router for ambiguous queries when cloud provider is active
        active_provider = config.get_active_provider()
        if active_provider != "local":
            llm_result = self._analyze_via_llm(raw_query)
            if llm_result:
                return llm_result

        return semantic_res

    def _analyze_via_llm(self, query: str) -> Optional[Dict[str, Any]]:
        system_prompt = (
            "You are the Query Understanding engine for an AI Restaurant Operations system. "
            "Analyze the user query and return valid JSON with:\n"
            "1. 'intent': One of ['complaint_analysis', 'root_cause', 'trend_analysis', 'severity_prioritization', 'recommendations', 'simple_retrieval'].\n"
            "2. 'aspect': One of ['Wait Time', 'Food Quality', 'Staff Behavior', 'Pricing', 'Food Safety / Illness'] or null if ambiguous.\n"
            "3. 'search_representation': Expanded semantic search query for BGE-M3 and BM25 review retrieval."
        )
        prompt = f"Analyze this user query: \"{query}\""
        try:
            resp = llm_engine.generate(prompt=prompt, system_prompt=system_prompt, json_mode=True)
            cleaned = re.sub(r"^```(?:json)?\s*", "", resp.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)
            aspect = data.get("aspect")
            intent = data.get("intent", "complaint_analysis")

            lq = query.lower()
            is_general_complaint = any(w in lq for w in [
                "complain", "complaint", "complaints", "criticism", "dissatisfaction",
                "customer issues", "problems", "negative feedback", "low rate", "low rating",
                "poor rating", "bad rating", "low stars", "bad review", "bad reviews",
                "getting low", "why low", "why getting"
            ])
            if (not aspect or aspect.lower() in ["none", "null", "general operations"]) and is_general_complaint:
                aspect = "General Operations"
                search_rep = "customer complaints about slow service, long wait times, bad food quality, cold food, rude staff, negative dining experience"
            else:
                search_rep = SEARCH_REPRESENTATIONS.get(aspect) or data.get("search_representation", query)

            return {
                "original_query": query,
                "retrieval_query": search_rep,
                "intent": intent,
                "aspect": aspect,
                "confidence": 0.95,
                "method": "llm"
            }
        except Exception as e:
            print(f"Notice: LLM query understanding skipped: {e}")
            return None

    def _analyze_via_semantic_router(self, query: str) -> Dict[str, Any]:
        intent = self._detect_intent(query)
        lq = query.lower()

        # Fast keyword taxonomy routing (instant 0.001s resolution)
        keyword_aspect = None
        if any(w in lq for w in ["rude", "attitude", "staff", "waiter", "waitress", "server", "hostess", "disrespect", "demeanor"]):
            keyword_aspect = "Staff Behavior"
        elif any(w in lq for w in ["cold", "taste", "undercooked", "raw", "bland", "salty", "burnt", "tasteless"]):
            keyword_aspect = "Food Quality"
        elif any(w in lq for w in ["slow", "wait", "waited", "waiting", "minutes", "hour", "hours", "delay", "queue", "line"]):
            keyword_aspect = "Wait Time"
        elif any(w in lq for w in ["expensive", "price", "pricing", "overpriced", "bill", "charge", "surcharge", "cost"]):
            keyword_aspect = "Pricing"
        elif any(w in lq for w in ["sick", "poison", "poisoning", "ill", "vomit", "nausea"]):
            keyword_aspect = "Food Safety / Illness"
        elif "food" in lq:
            keyword_aspect = "Food Quality"

        if keyword_aspect:
            return {
                "original_query": query,
                "retrieval_query": SEARCH_REPRESENTATIONS.get(keyword_aspect, query),
                "intent": intent,
                "aspect": keyword_aspect,
                "confidence": 0.98,
                "method": "fast_taxonomy_router"
            }

        best_aspect = None
        best_score = -1.0

        if self.embedder is not None and self.aspect_vectors:
            try:
                q_vec = self.embedder.encode(query, normalize_embeddings=True, show_progress_bar=False)
                for name, vecs in self.aspect_vectors.items():
                    sims = np.dot(vecs, q_vec)
                    max_sim = float(np.max(sims))
                    if max_sim > best_score:
                        best_score = max_sim
                        best_aspect = name
            except Exception as e:
                print(f"Notice: Semantic embedding comparison error: {e}")

        # Strict confidence threshold
        if best_score >= 0.50:
            aspect = best_aspect
            confidence = round(best_score, 4)
        else:
            aspect = None
            confidence = round(best_score, 4) if best_score > 0 else 0.0

        # Construct Search Representation (retrieval_query)
        if aspect and aspect in SEARCH_REPRESENTATIONS:
            search_representation = SEARCH_REPRESENTATIONS[aspect]
        else:
            lq = query.lower()
            if any(w in lq for w in [
                "complain", "complaint", "complaints", "criticism", "dissatisfaction",
                "customer issues", "problems", "negative feedback", "low rate", "low rating",
                "poor rating", "bad rating", "low stars", "bad review", "bad reviews",
                "getting low", "why low", "why getting"
            ]):
                search_representation = "customer complaints about slow service, long wait times, bad food quality, cold food, rude staff, negative dining experience"
                aspect = "General Operations"
                confidence = 0.88
            else:
                # When ambiguous or no aspect detected, preserve raw query
                search_representation = query

        return {
            "original_query": query,
            "retrieval_query": search_representation,
            "intent": intent,
            "aspect": aspect,
            "confidence": confidence,
            "method": "semantic_router"
        }

# Global Singleton
query_understanding = QueryUnderstandingEngine()
