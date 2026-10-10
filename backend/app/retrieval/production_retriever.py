import os
import re
import pickle
import threading
import logging
import numpy as np
from typing import List, Dict, Any, Optional

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BM25_INDEX_FILE = os.path.join(DATA_DIR, "bm25_index.pkl")
COLLECTION_NAME = "yelp_reviews_bge_m3"
VECTOR_DIM = 1024

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", 
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", 
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", 
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", 
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", 
    "with", "would", "you", "your", "yours", "yourself", "yourselves", "tell", "show", "what"
}

class ProductionRetriever:
    """
    Production Hybrid Retrieval Engine:
    1. BGE-M3 (1024-dim) Dense Retrieval -> Top 50 from Qdrant
    2. Persistent BM25 Lexical Retrieval -> Top 50 from full corpus
    3. Reciprocal Rank Fusion (RRF, k=60)
    4. Exact Deduplication
    5. BGE Reranker v2 M3 Cross-Encoder -> Top 8-10 candidates
    6. Strict Evidence Gate (negation filter, affirmative validation, no evidence -> no claim)
    """
    def __init__(
        self,
        qdrant_url: str = "http://127.0.0.1:6333",
        collection_name: str = COLLECTION_NAME
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.qdrant_client: Optional[QdrantClient] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None
        self.bm25_model: Optional[BM25Okapi] = None
        self.bm25_review_ids: List[str] = []
        self.bm25_id_to_idx: Dict[str, int] = {}
        self._review_cache: Dict[str, Dict[str, Any]] = {}
        self._query_emb_cache: Dict[str, np.ndarray] = {}
        self._reranker_lock = threading.Lock()
        self._initialize()

    def _initialize(self):
        # 1. Connect to Qdrant
        try:
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
        except Exception as e:
            print(f"Warning: Could not connect to Qdrant at {self.qdrant_url}: {e}")
            self.qdrant_client = None

        # Determine optimal hardware acceleration device (MPS on Apple Silicon, CUDA on Nvidia, CPU fallback)
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Load BGE-M3 Embedder
        try:
            self.embedder = SentenceTransformer("BAAI/bge-m3", device=device, local_files_only=True)
        except Exception:
            try:
                self.embedder = SentenceTransformer("BAAI/bge-m3", device=device)
            except Exception as e:
                print(f"Warning: Could not load BGE-M3: {e}")
                self.embedder = None

        # 3. Load BGE Reranker v2 M3
        try:
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, local_files_only=True)
        except Exception:
            try:
                self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
            except Exception as e:
                print(f"Warning: Could not load BGE Reranker: {e}")
                self.reranker = None

        # 4. Load persistent BM25 index
        self._load_bm25_index()

        # 5. Pre-warm in-memory review cache
        try:
            from backend.app.dataset.yelp_dataset import dataset_manager
            self._review_cache = {str(r.get("review_id")): r for r in dataset_manager.reviews if r.get("review_id")}
        except Exception:
            self._review_cache = {}

        # 6. Pre-warm MPS models so first user query runs in sub-second time
        if self.embedder is not None:
            try:
                _ = self.embedder.encode("warmup query", convert_to_numpy=True)
            except Exception:
                pass
        if self.reranker is not None:
            try:
                _ = self.reranker.predict([("warmup query", "warmup document")])
            except Exception:
                pass

    def _load_bm25_index(self):
        if os.path.exists(BM25_INDEX_FILE):
            try:
                with open(BM25_INDEX_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.bm25_model = data.get("bm25")
                    self.bm25_review_ids = data.get("review_ids", [])
                    self.bm25_id_to_idx = {rid: i for i, rid in enumerate(self.bm25_review_ids)}
            except Exception as e:
                print(f"Warning: Could not load BM25 index from {BM25_INDEX_FILE}: {e}")
                self.bm25_model = None
                self.bm25_id_to_idx = {}

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        filtered = [t for t in tokens if t not in STOP_WORDS]
        return filtered if filtered else tokens

    def _check_evidence_affirmation(self, query: str, review_text: str, stars: float = 1.0) -> bool:
        """
        Validates affirmative evidence.
        Rejects negated occurrences (e.g., 'no one got sick', 'never had an issue')
        and non-illness idioms (e.g., 'never get sick of').
        """
        lower_q = query.lower()
        lower_t = review_text.lower()

        # Specifically for illness/poisoning questions
        illness_terms = ["sick", "poisoning", "vomit", "nausea", "stomach ache", "food poisoning"]
        is_illness_q = any(w in lower_q for w in illness_terms) or bool(re.search(r"\b(ill|illness)\b", lower_q))
        if is_illness_q:
            # First remove non-illness idioms like 'never get sick of', 'sick of'
            cleaned = re.sub(r"\b(never|don'?t|can'?t|won'?t|hardly)\s+get\s+sick\s+of\b", " ", lower_t)
            cleaned = re.sub(r"\bsick\s+of\b", " ", cleaned)

            # Check for negated claims and strip them out
            negation_patterns = [
                r"\bno\s+one\s+got\s+sick\b",
                r"\bnobody\s+got\s+sick\b",
                r"\bnever\s+got\s+sick\b",
                r"\bdid\s+not\s+get\s+sick\b",
                r"\bdidn't\s+get\s+sick\b",
                r"\bwasn't\s+sick\b",
                r"\bnot\s+sick\b",
                r"\bno\s+food\s+poisoning\b",
                r"\bwithout\s+getting\s+sick\b",
                r"\bno\s+one\s+became\s+ill\b",
                r"\bnobody\s+became\s+ill\b",
                r"\bnever\s+had\s+food\s+poisoning\b"
            ]
            has_negation = False
            for pat in negation_patterns:
                if re.search(pat, cleaned):
                    has_negation = True
                    cleaned = re.sub(pat, " ", cleaned)

            # Check affirmative expressions in the remaining cleaned text
            affirmative_patterns = [
                r"\bgot\s+sick\b",
                r"\bfood\s+poisoning\b",
                r"\bvomit(ed|ing)?\b",
                r"\bthrew\s+up\b",
                r"\bnausea\b",
                r"\bdiarrhea\b",
                r"\bviolent\s+illness\b",
                r"\bbecame\s+ill\b",
                r"\bgot\s+ill\b",
                r"\bwent\s+to\s+the\s+(er|hospital)\b",
                r"\bstomach\s+(ache|pain|cramp|cramps)\b"
            ]
            has_affirmative = any(re.search(p, cleaned) for p in affirmative_patterns)

            if has_negation and not has_affirmative:
                return False

            if not has_affirmative:
                return False

        # For complaint/critique questions: reject purely positive reviews with no complaint words
        is_complaint_q = any(w in lower_q for w in [
            "complain", "complaints", "bad", "slow", "rude", "poor", "issue", "problem",
            "disappoint", "terrible", "awful", "cold", "undercooked", "delay", "waiting",
            "overprice", "dirty", "worst", "unacceptable", "criticism", "hate"
        ])
        if is_complaint_q and stars >= 4.0:
            critique_terms = [
                "slow", "wait", "waited", "delay", "rude", "attitude", "ignore", "cold", "raw",
                "bad", "terrible", "awful", "horrible", "overpriced", "expensive", "wrong",
                "disappoint", "dirty", "problem", "issue", "complaint", "mistake", "poor"
            ]
            has_critique = any(w in lower_t for w in critique_terms)
            if not has_critique:
                return False

        return True

    def _safe_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Safely reranks candidates with BGE Reranker v2 M3 Cross-Encoder.
        Guarantees:
        - Thread safety via Lock to avoid multi-threading memory corruption
        - Input sanitization (None/empty text handling)
        - Safe extraction of 1D/2D/scalar numpy/tensor outputs
        - Monotonic descending sorting by reranker_score
        - Clean fallback to rrf_score on model failure without throwing exceptions
        """
        if not candidates:
            return []

        # If reranker model is not loaded, fallback gracefully
        if self.reranker is None:
            for item in candidates:
                item["reranker_score"] = float(item.get("rrf_score", 0.0))
            return sorted(candidates, key=lambda x: x.get("reranker_score", 0.0), reverse=True)

        try:
            # Sanitize text with compact length for rapid inference
            pairs = []
            for item in candidates:
                text_val = item.get("text")
                clean_text = str(text_val or "")[:256]
                pairs.append((query, clean_text))

            with self._reranker_lock:
                raw_scores = self.reranker.predict(pairs, max_length=160, batch_size=len(pairs))

            # Normalize raw_scores to a 1D list of floats
            norm_scores = []
            if isinstance(raw_scores, (int, float, np.floating, np.integer)):
                norm_scores = [float(raw_scores)]
            elif hasattr(raw_scores, "ndim"):
                if raw_scores.ndim == 0:
                    norm_scores = [float(raw_scores.item())]
                elif raw_scores.ndim == 1:
                    norm_scores = [float(s) for s in raw_scores]
                elif raw_scores.ndim == 2:
                    if raw_scores.shape[1] > 1:
                        norm_scores = [float(row[1]) for row in raw_scores]
                    else:
                        norm_scores = [float(row[0]) for row in raw_scores]
                else:
                    norm_scores = [float(s) for s in raw_scores.flatten()]
            elif isinstance(raw_scores, (list, tuple)):
                for s in raw_scores:
                    if isinstance(s, (list, tuple, np.ndarray)):
                        norm_scores.append(float(s[0]))
                    else:
                        norm_scores.append(float(s))
            else:
                norm_scores = [float(s) for s in raw_scores]

            if len(norm_scores) != len(candidates):
                raise ValueError(f"Score count mismatch: got {len(norm_scores)}, expected {len(candidates)}")

            for item, score in zip(candidates, norm_scores):
                item["reranker_score"] = round(float(score), 6)

            candidates.sort(key=lambda x: x.get("reranker_score", 0.0), reverse=True)
            return candidates

        except Exception as e:
            logging.warning(f"Error running BGE Reranker: {e}")
            for item in candidates:
                item["reranker_score"] = float(item.get("rrf_score", 0.0))
            candidates.sort(key=lambda x: x.get("reranker_score", 0.0), reverse=True)
            return candidates

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 50,
        bm25_top_k: int = 50,
        final_top_k: int = 8,
        k_rrf: int = 60,
        business_id: Optional[str] = None,
        min_stars: Optional[int] = None,
        max_stars: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes production multi-stage hybrid retrieval:
        Dense Top 50 + BM25 Top 50 -> RRF -> Dedup -> BGE Reranker -> Evidence Gate.
        """
        import time
        t_ret_start = time.time()
        diagnostics = {
            "query": query,
            "dense_count": 0,
            "bm25_count": 0,
            "fused_count": 0,
            "deduplicated_count": 0,
            "reranked_count": 0,
            "final_count": 0,
            "evidence_gate_status": "insufficient_evidence",
            "evidence_gate_notes": ""
        }

        if not query.strip():
            return {"results": [], "diagnostics": diagnostics}

        dense_candidates = {}
        bm25_candidates = {}
        doc_store = {}
        target_restaurant_name = None

        # If a specific restaurant is selected, inject its review records directly from PostgreSQL
        if business_id and business_id.lower() != "all":
            b_target = business_id.strip()
            try:
                import psycopg2
                conn = psycopg2.connect(host="127.0.0.1", port=5432, dbname="postgres", user="bharathyadav", password="", connect_timeout=1)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT r.review_id, r.business_id, COALESCE(b.name, 'Unknown Restaurant'), r.stars, r.date, r.text, COALESCE(b.city, ''), COALESCE(b.state, '')
                        FROM yelp_reviews r
                        LEFT JOIN yelp_businesses b ON r.business_id = b.business_id
                        WHERE LOWER(b.name) = %s OR LOWER(r.business_id) = %s
                        ORDER BY r.date DESC;
                    """, (b_target.lower(), b_target.lower()))
                    biz_rows = cur.fetchall()
                conn.close()
                if biz_rows:
                    target_restaurant_name = biz_rows[0][2]
                    for row in biz_rows:
                        r_id = row[0]
                        doc_store[r_id] = {
                            "review_id": row[0],
                            "business_id": row[1],
                            "business_name": row[2],
                            "stars": float(row[3] or 1.0),
                            "date": str(row[4] or ""),
                            "text": row[5] or "",
                            "city": row[6] or "",
                            "state": row[7] or ""
                        }
                else:
                    target_restaurant_name = business_id
            except Exception as e:
                target_restaurant_name = business_id

            if not doc_store:
                from backend.app.dataset.yelp_dataset import dataset_manager
                for r in dataset_manager.reviews:
                    if (r.get("restaurant_name", "").lower() == b_target.lower() or 
                        r.get("business_id", "").lower() == b_target.lower()):
                        target_restaurant_name = r.get("restaurant_name", b_target)
                        doc_store[r["review_id"]] = {
                            "review_id": r["review_id"],
                            "business_id": r.get("business_id", ""),
                            "business_name": r.get("restaurant_name", "Unknown Restaurant"),
                            "stars": float(r.get("stars", 1.0)),
                            "date": str(r.get("date", "")),
                            "text": r.get("text", ""),
                            "city": r.get("city", ""),
                            "state": r.get("state", "")
                        }

        # ----------------------------------------------------
        # Stage 1: Dense Retrieval via BGE-M3 & Qdrant (Top 50)
        # ----------------------------------------------------
        if self.embedder is not None and self.qdrant_client is not None:
            try:
                if query in self._query_emb_cache:
                    q_vec = self._query_emb_cache[query]
                else:
                    q_vec = self.embedder.encode(query, convert_to_numpy=True)
                    if len(self._query_emb_cache) > 200:
                        self._query_emb_cache.pop(next(iter(self._query_emb_cache)))
                    self._query_emb_cache[query] = q_vec
                search_res = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=q_vec.tolist(),
                    limit=dense_top_k,
                    with_payload=True
                )
                for rank, hit in enumerate(search_res):
                    payload = hit.payload or {}
                    r_id = payload.get("review_id") or payload.get("id") or str(hit.id)
                    dense_candidates[r_id] = {
                        "rank": rank + 1,
                        "score": float(hit.score),
                        "payload": payload
                    }
                    if r_id not in doc_store:
                        doc_store[r_id] = payload
                diagnostics["dense_count"] = len(dense_candidates)
            except Exception as e:
                print(f"Error querying Qdrant: {e}")

        # ----------------------------------------------------
        # Stage 2: Lexical Retrieval via BM25 (Top 50)
        # ----------------------------------------------------
        bm25_scores = None
        if self.bm25_model is not None and self.bm25_review_ids:
            try:
                q_tokens = self._tokenize(query)
                lq = query.lower()
                is_complaint_inquiry = any(w in lq for w in [
                    "complain", "complaint", "complaints", "bad", "poor", "slow", "rude", "issue", "problem", "disappoint"
                ])
                if not q_tokens or (is_complaint_inquiry and all(t in STOP_WORDS for t in re.findall(r"\b[a-zA-Z]{3,}\b", lq))):
                    q_tokens = ["slow", "service", "wait", "food", "cold", "rude", "staff", "bad", "terrible"]

                if q_tokens:
                    bm25_scores = self.bm25_model.get_scores(q_tokens)
                    top_indices = np.argsort(-bm25_scores)[:bm25_top_k]
                    for rank, idx in enumerate(top_indices):
                        score = float(bm25_scores[idx])
                        if score > 0.01:
                            r_id = self.bm25_review_ids[idx]
                            bm25_candidates[r_id] = {
                                "rank": rank + 1,
                                "score": score
                            }

                    if not bm25_candidates and is_complaint_inquiry:
                        fallback_tokens = ["slow", "service", "wait", "food", "cold", "rude", "staff", "bad", "terrible"]
                        bm25_scores = self.bm25_model.get_scores(fallback_tokens)
                        top_indices = np.argsort(-bm25_scores)[:bm25_top_k]
                        for rank, idx in enumerate(top_indices):
                            score = float(bm25_scores[idx])
                            if score > 0.01:
                                r_id = self.bm25_review_ids[idx]
                                bm25_candidates[r_id] = {
                                    "rank": rank + 1,
                                    "score": score
                                }

                diagnostics["bm25_count"] = len(bm25_candidates)
            except Exception as e:
                print(f"Error evaluating BM25: {e}")

        # If a single restaurant is selected, score its reviews with BM25 if available
        if target_restaurant_name and doc_store:
            for r_id in list(doc_store.keys()):
                if r_id not in bm25_candidates and bm25_scores is not None and r_id in self.bm25_id_to_idx:
                    bm25_candidates[r_id] = {
                        "rank": 50,
                        "score": float(bm25_scores[self.bm25_id_to_idx[r_id]])
                    }

        # ----------------------------------------------------
        # Stage 3: Reciprocal Rank Fusion (RRF) & Deduplication
        # ----------------------------------------------------
        all_ids = set(dense_candidates.keys()).union(set(bm25_candidates.keys())).union(set(doc_store.keys()))
        diagnostics["deduplicated_count"] = len(all_ids)

        # Hydrate document payloads for candidates missing full text (in-memory cache first for instant speed)
        missing_ids = [r_id for r_id in all_ids if r_id not in doc_store or not doc_store[r_id].get("text")]
        if missing_ids and self._review_cache:
            for r_id in list(missing_ids):
                rc = self._review_cache.get(r_id) or self._review_cache.get(str(r_id))
                if rc:
                    doc_store[r_id] = {
                        "review_id": str(rc.get("review_id", r_id)),
                        "business_id": str(rc.get("business_id", "")),
                        "business_name": str(rc.get("restaurant_name", "Unknown Restaurant")),
                        "stars": float(rc.get("stars", 1.0)),
                        "date": str(rc.get("date", "")),
                        "text": str(rc.get("text", "")),
                        "city": str(rc.get("city", "")),
                        "state": str(rc.get("state", ""))
                    }

        # Secondary hydration from PostgreSQL only for remaining missing candidates
        still_missing = [r_id for r_id in all_ids if r_id not in doc_store or not doc_store[r_id].get("text")]
        if still_missing:
            try:
                import psycopg2
                conn = psycopg2.connect(host="127.0.0.1", port=5432, dbname="postgres", user="bharathyadav", password="", connect_timeout=1)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT r.review_id, r.business_id, COALESCE(b.name, 'Unknown Restaurant'), r.stars, r.date, r.text, COALESCE(b.city, ''), COALESCE(b.state, '')
                        FROM yelp_reviews r
                        LEFT JOIN yelp_businesses b ON r.business_id = b.business_id
                        WHERE r.review_id = ANY(%s);
                    """, (list(still_missing),))
                    for row in cur.fetchall():
                        doc_store[row[0]] = {
                            "review_id": row[0],
                            "business_id": row[1],
                            "business_name": row[2],
                            "stars": float(row[3] or 1.0),
                            "date": str(row[4] or ""),
                            "text": row[5] or "",
                            "city": row[6] or "",
                            "state": row[7] or ""
                        }
                conn.close()
            except Exception:
                pass

        fused_items = []
        for r_id in all_ids:
            dense_info = dense_candidates.get(r_id)
            bm25_info = bm25_candidates.get(r_id)

            r_dense = dense_info["rank"] if dense_info else 1000
            r_bm25 = bm25_info["rank"] if bm25_info else 1000

            rrf_score = (1.0 / (k_rrf + r_dense)) + (1.0 / (k_rrf + r_bm25))
            
            payload = doc_store.get(r_id)
            if not payload:
                # Fallback empty payload
                payload = {"review_id": r_id, "text": "", "business_name": "Unknown", "stars": 1, "date": "", "city": "", "state": ""}

            b_name = payload.get("business_name") or payload.get("restaurant_name", "Unknown Restaurant")
            fused_items.append({
                "review_id": r_id,
                "business": b_name,
                "restaurant_name": b_name,
                "business_id": payload.get("business_id", ""),
                "stars": payload.get("stars", 1.0),
                "date": payload.get("date", ""),
                "text": payload.get("text", ""),
                "city": payload.get("city", ""),
                "state": payload.get("state", ""),
                "rrf_score": round(float(rrf_score), 6),
                "dense_rank": r_dense if r_dense < 1000 else None,
                "bm25_rank": r_bm25 if r_bm25 < 1000 else None
            })

        # Apply metadata filters if specified
        if business_id and business_id.lower() != "all":
            b_target = business_id.lower()
            fused_items = [
                item for item in fused_items
                if item["business_id"].lower() == b_target or item["business"].lower() == b_target
            ]
        if min_stars is not None:
            fused_items = [item for item in fused_items if item["stars"] >= min_stars]
        if max_stars is not None:
            fused_items = [item for item in fused_items if item["stars"] <= max_stars]

        fused_items.sort(key=lambda x: x["rrf_score"], reverse=True)
        diagnostics["fused_count"] = len(fused_items)

        # Candidate pool for reranker (top 6 fused items for fast sub-second inference)
        pool_size = min(max(final_top_k + 1, 6), len(fused_items))
        rerank_candidates = fused_items[:pool_size]

        # ----------------------------------------------------
        # Stage 4: BGE Reranker v2 M3
        # ----------------------------------------------------
        rerank_candidates = self._safe_rerank(query, rerank_candidates)
        diagnostics["reranked_count"] = len(rerank_candidates)

        # ----------------------------------------------------
        # Stage 5: Evidence Gate
        # ----------------------------------------------------
        passed_evidence = []
        for item in rerank_candidates:
            # Check affirmative evidence
            if self._check_evidence_affirmation(query, item["text"], float(item.get("stars", 1.0))):
                # Relevance filter: discard negligible background noise (< 0.001)
                if item.get("reranker_score", 1.0) >= 0.001:
                    passed_evidence.append(item)
            if len(passed_evidence) >= final_top_k:
                break

        resolved_name = target_restaurant_name or (business_id if business_id and business_id.lower() != "all" else "All Restaurants")
        diagnostics["target_restaurant"] = resolved_name

        if not passed_evidence:
            diagnostics["evidence_gate_status"] = "insufficient_evidence"
            if resolved_name != "All Restaurants":
                diagnostics["no_complaint_notice"] = f"No customer complaints regarding this issue were found for {resolved_name}."
            else:
                diagnostics["no_complaint_notice"] = "No customer complaints regarding this issue were found across all indexed reviews."
            diagnostics["evidence_gate_notes"] = diagnostics["no_complaint_notice"]
            diagnostics["final_count"] = 0
            return {
                "results": [],
                "diagnostics": diagnostics
            }

        diagnostics["evidence_gate_status"] = "sufficient_evidence"
        diagnostics["evidence_gate_notes"] = f"Validated {len(passed_evidence)} affirmative review evidence records."
        diagnostics["no_complaint_notice"] = None
        diagnostics["final_count"] = len(passed_evidence)
        logging.info(f"[ProductionRetriever Latency] Total={time.time()-t_ret_start:.3f}s (candidates={len(fused_items)}, passed_evidence={len(passed_evidence)})")

        return {
            "results": passed_evidence,
            "diagnostics": diagnostics
        }

# Global Singleton
production_retriever = ProductionRetriever()
