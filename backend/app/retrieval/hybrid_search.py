import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import re
import math
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from backend.app.dataset.yelp_dataset import dataset_manager

STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
    "complaint", "complaints", "report", "reported", "reporting",
    "issue", "issues", "problem", "problems", "what", "are", "about",
    "thing", "things", "review", "reviews", "customer", "customers",
    "restaurant", "tell", "explain", "give", "giving", "does", "did",
    "show", "similar", "associated", "with", "them"
})

CATEGORY_INTENT_MAP = {
    "Food Quality": [
        "taste", "tasted", "tasting", "flavor", "flavors", "salty", "bland", "sour",
        "bitter", "inedible", "cold", "undercooked", "raw", "burnt", "charred", "stale",
        "rubbery", "food", "dish", "dishes", "meal", "pasta", "steak", "chicken", "soup",
        "curry", "ramen", "burger", "seafood", "cooking", "seasoning", "sauce"
    ],
    "Wait Time": [
        "slow", "late", "wait", "waited", "waiting", "delay", "delayed", "delays",
        "minutes", "minute", "hour", "hours", "queue", "line", "forever", "seating",
        "timing", "ticket", "backlog", "pacing"
    ],
    "Staff & Service": [
        "rude", "attitude", "ignored", "server", "servers", "waiter", "waiters",
        "waitress", "hostess", "manager", "staff", "service", "hospitality", "demeanor",
        "unfriendly", "disrespectful", "argument", "inattentive"
    ],
    "Cleanliness & Hygiene": [
        "dirty", "filthy", "clean", "cleanliness", "restroom", "toilet", "hair",
        "staple", "poisoning", "sick", "vomit", "hygiene", "sanitation", "soap",
        "disgusting", "roach", "flies", "fly"
    ],
    "Order Accuracy": [
        "wrong", "missing", "forgot", "forgotten", "allergy", "allergic", "gluten",
        "celiac", "substitute", "substituted", "incorrect", "order", "sides"
    ],
    "Pricing & Value": [
        "price", "prices", "overpriced", "expensive", "surcharge", "fee", "fees",
        "bill", "receipt", "check", "gratuity", "double-charge", "portion", "rip off"
    ],
    "Ambience & Noise": [
        "loud", "noise", "noisy", "music", "ac", "air condition", "hot", "freezing",
        "dark", "lighting", "seating", "table", "ambience", "atmosphere", "drafty"
    ]
}

class HybridRetriever:
    """
    Precision Hybrid Search Engine combining:
    1. Dense semantic search via local SentenceTransformers (all-MiniLM-L6-v2)
    2. Intent-filtered sparse keyword search via BM25 (Okapi)
    3. Category intent alignment and relevance calibration
    4. Multi-attribute metadata filtering (restaurant, category, city, state, rating)
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = None
        self.corpus_reviews: List[Dict[str, Any]] = []
        self.corpus_texts: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._load_embedder()
        self.index_reviews()
        self._initialized = True

    def _load_embedder(self):
        try:
            self.embedder = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Warning: Could not load {self.model_name}: {e}")
            self.embedder = None

    @staticmethod
    def _tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenizes text and removes irrelevant stop words for high BM25 precision."""
        raw_tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        if remove_stopwords:
            filtered = [t for t in raw_tokens if t not in STOP_WORDS]
            return filtered if filtered else raw_tokens
        return raw_tokens

    def index_reviews(self):
        """Indexes all reviews with enriched operational category and aspect markers."""
        self.corpus_reviews = dataset_manager.get_all_reviews()
        
        # Enriched semantic document representation
        self.corpus_texts = [
            f"Restaurant: {r.get('restaurant_name', '')} | Operational Category: {r.get('category', '')} | Issue Aspect: {r.get('aspect', '')} | Customer Feedback: {r.get('text', '')}"
            for r in self.corpus_reviews
        ]

        # BM25 tokenization with stopword filtering
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus_texts]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Dense embedding indexing on demand
        if self.embedder is not None and self.corpus_texts and self.corpus_embeddings is None:
            raw_embs = self.embedder.encode(self.corpus_texts, convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            self.corpus_embeddings = raw_embs / norms

    def _detect_query_intent_category(self, query: str) -> Optional[str]:
        """Identifies primary operational category intended by query keywords."""
        q_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
        best_cat = None
        max_matches = 0

        for cat, keywords in CATEGORY_INTENT_MAP.items():
            matches = len(q_tokens.intersection(set(keywords)))
            if matches > max_matches:
                max_matches = matches
                best_cat = cat

        return best_cat if max_matches > 0 else None

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        business_id: Optional[str] = None,
        category: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        min_stars: Optional[int] = None,
        max_stars: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search with metadata filtering and intent alignment.
        Delegates to production_retriever (Qdrant BGE-M3 + BM25 + RRF + BGE Reranker + Evidence Gate)
        when production data is present.
        """
        try:
            from backend.app.retrieval.production_retriever import production_retriever
            if production_retriever is not None and production_retriever.qdrant_client is not None and production_retriever.bm25_model is not None:
                # Apply Query Understanding layer before retrieval
                retrieval_q = query
                try:
                    from backend.app.query_understanding.query_analyzer import query_understanding
                    qu = query_understanding.analyze(query)
                    retrieval_q = qu.get("retrieval_query", query)
                except Exception:
                    pass

                prod_res = production_retriever.retrieve(
                    query=retrieval_q,
                    dense_top_k=50,
                    bm25_top_k=50,
                    final_top_k=top_k,
                    business_id=business_id,
                    min_stars=min_stars,
                    max_stars=max_stars
                )
                evidence = prod_res.get("results", [])
                diag = prod_res.get("diagnostics", {})
                self.last_diagnostics = diag
                
                # If Evidence Gate determined insufficient evidence or negated claims
                if diag.get("evidence_gate_status") == "insufficient_evidence":
                    return []
                
                def _safe_float(val, fallback=0.0):
                    if val is None:
                        return fallback
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return fallback

                if evidence:
                    results = []
                    for r in evidence:
                        h_score = _safe_float(r.get("reranker_score"), _safe_float(r.get("rrf_score"), 0.0))
                        results.append({
                            "review_id": str(r.get("review_id", "")),
                            "business_id": str(r.get("business_id", "")),
                            "restaurant_name": str(r.get("business", "Unknown Restaurant")),
                            "business": str(r.get("business", "Unknown Restaurant")),
                            "stars": _safe_float(r.get("stars"), 1.0),
                            "date": str(r.get("date", "")),
                            "text": str(r.get("text", "")),
                            "city": str(r.get("city", "")),
                            "state": str(r.get("state", "")),
                            "hybrid_score": round(h_score, 4),
                            "semantic_score": round(_safe_float(r.get("dense_rank"), 1.0), 4),
                            "bm25_score": round(_safe_float(r.get("bm25_rank"), 1.0), 4),
                            "rrf_score": round(_safe_float(r.get("rrf_score"), 0.0), 6),
                            "category": "Customer Review",
                            "aspect": "Real Yelp Evidence"
                        })
                    return results
        except Exception as e:
            print(f"Warning: Production retriever error ({e}), falling back to local retriever.")

        self._ensure_initialized()

        n_docs = len(self.corpus_reviews)
        if n_docs == 0:
            return []

        # 1. Metadata Filter Mask
        valid_indices = []
        for i, r in enumerate(self.corpus_reviews):
            if business_id and business_id.lower() != "all":
                b_match = (
                    r.get("business_id", "").lower() == business_id.lower() or
                    r.get("restaurant_name", "").lower() == business_id.lower()
                )
                if not b_match:
                    continue
            if category and category.lower() != "all":
                b_info = dataset_manager.get_business(r["business_id"]) or {}
                b_cats = b_info.get("categories", "").lower()
                r_cat = r.get("category", "").lower()
                if category.lower() not in b_cats and category.lower() not in r_cat:
                    continue
            if city and city.lower() != "all" and r.get("city", "").lower() != city.lower():
                continue
            if state and state.lower() != "all" and r.get("state", "").lower() != state.lower():
                continue
            if min_stars is not None and r.get("stars", 5) < min_stars:
                continue
            if max_stars is not None and r.get("stars", 1) > max_stars:
                continue
            if start_date and r.get("date", "")[:10] < start_date:
                continue
            if end_date and r.get("date", "")[:10] > end_date:
                continue
            valid_indices.append(i)

        if not valid_indices:
            return []

        # 2. BM25 scoring with stopword-free tokens
        bm25_scores = np.zeros(n_docs)
        if self.bm25:
            tokens = self._tokenize(query, remove_stopwords=True)
            if tokens:
                raw_bm25 = np.array(self.bm25.get_scores(tokens))
                filtered_bm25 = raw_bm25[valid_indices]
                min_b, max_b = filtered_bm25.min(), filtered_bm25.max()
                if max_b > min_b:
                    bm25_scores[valid_indices] = (filtered_bm25 - min_b) / (max_b - min_b)
                elif max_b > 0:
                    bm25_scores[valid_indices] = 1.0

        # 3. Dense semantic vector scoring
        semantic_scores = np.zeros(n_docs)
        if self.embedder is not None and self.corpus_embeddings is not None:
            # Query embedding
            q_emb = self.embedder.encode(query, convert_to_numpy=True, show_progress_bar=False)
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 0:
                q_emb = q_emb / q_norm
                sims = np.dot(self.corpus_embeddings, q_emb)
                sims_scaled = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
                semantic_scores[valid_indices] = sims_scaled[valid_indices]

        # 4. Intent Category Alignment & Relevance Calibration
        detected_category = self._detect_query_intent_category(query)

        # 5. Composite Ranking
        dense_order = np.argsort(-semantic_scores[valid_indices])
        bm25_order = np.argsort(-bm25_scores[valid_indices])
        dense_rank_map = {valid_indices[idx]: rank + 1 for rank, idx in enumerate(dense_order)}
        bm25_rank_map = {valid_indices[idx]: rank + 1 for rank, idx in enumerate(bm25_order)}

        results = []
        k_rrf = 60
        for idx in valid_indices:
            r = self.corpus_reviews[idx]
            s_score = float(semantic_scores[idx])
            b_score = float(bm25_scores[idx])
            
            # Base fusion
            hybrid_score = (alpha * s_score) + ((1.0 - alpha) * b_score)

            # Intent alignment boost/penalty
            if detected_category:
                r_cat = r.get("category", "")
                r_aspect = r.get("aspect", "")
                if r_cat == detected_category or detected_category in r_aspect:
                    # Strong boost for matching domain
                    hybrid_score = min(1.0, hybrid_score * 1.30 + 0.15)
                else:
                    # Penalty for non-matching domain when explicit domain requested
                    hybrid_score = hybrid_score * 0.40

            r_dense = dense_rank_map.get(idx, n_docs)
            r_bm25 = bm25_rank_map.get(idx, n_docs)
            rrf_score = (1.0 / (k_rrf + r_dense)) + (1.0 / (k_rrf + r_bm25))

            review_data = dict(r)
            review_data.update({
                "hybrid_score": round(float(hybrid_score), 4),
                "semantic_score": round(float(s_score), 4),
                "bm25_score": round(float(b_score), 4),
                "rrf_score": round(float(rrf_score), 6)
            })
            results.append(review_data)

        # Sort by calibrated hybrid_score descending
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]

# Global singleton
hybrid_retriever = HybridRetriever()
