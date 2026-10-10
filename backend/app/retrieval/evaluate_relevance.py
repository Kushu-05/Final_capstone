import os
import sys
import json
from typing import List, Dict, Any, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from backend.app.query_understanding.query_analyzer import query_understanding
from backend.app.retrieval.production_retriever import production_retriever

# Load ground-truth Yelp reviews annotated with operational categories & aspects
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "yelp_reviews.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    ALL_REVIEWS = json.load(f)

REVIEW_MAP = {r["review_id"]: r for r in ALL_REVIEWS}

# Benchmark query suite across core restaurant operational friction categories
BENCHMARK_CASES = [
    {
        "id": "Q1_WAIT_TIME",
        "query": "What are the complaints about long waiting times?",
        "target_category": "Service Speed",
        "target_aspect": "Kitchen Ticket & Seating Pacing"
    },
    {
        "id": "Q2_FOOD_QUALITY",
        "query": "What are the biggest food quality complaints?",
        "target_category": "Food Quality",
        "target_aspect": "Recipe Execution & Thermal Control"
    },
    {
        "id": "Q3_STAFF",
        "query": "What are customers saying about staff behavior and rudeness?",
        "target_category": "Staff Demeanor",
        "target_aspect": "Hospitality & Guest Interaction"
    },
    {
        "id": "Q4_PRICING",
        "query": "What are the main pricing complaints and overcharging issues?",
        "target_category": "Pricing & Value",
        "target_aspect": "Transparent Billing & Fair Value"
    },
    {
        "id": "Q5_ILLNESS",
        "query": "Did any customers report getting sick or food poisoning?",
        "target_category": "Food Safety & Hygiene",
        "target_aspect": "Contamination & Health Risk"
    }
]

def run_benchmark(k: int = 8) -> Dict[str, Any]:
    print("=" * 80)
    print("RESTAURANTOPS AI — RETRIEVAL RELEVANCE BENCHMARK (GROUND-TRUTH YELP DATA)")
    print(f"Total Reference Reviews in Dataset: {len(ALL_REVIEWS)}")
    print(f"Evaluation Metrics: Precision@{k}, Recall@{k}, MRR (Mean Reciprocal Rank)")
    print("=" * 80)

    overall_metrics = {
        f"precision@{k}": [],
        f"recall@{k}": [],
        "reciprocal_rank": [],
        "hit_rate": []
    }

    per_query_results = []

    for case in BENCHMARK_CASES:
        q_id = case["id"]
        q_text = case["query"]
        tgt_cat = case["target_category"]
        tgt_asp = case["target_aspect"]

        # Ground truth relevant review IDs for this complaint category
        # Relevant if review belongs to the target category or target aspect
        gt_ids = {
            r["review_id"] for r in ALL_REVIEWS 
            if r.get("category") == tgt_cat or r.get("aspect") == tgt_asp
        }
        total_gt = len(gt_ids)

        print(f"\nEvaluating {q_id}: \"{q_text}\"")
        print(f"  Target Category: {tgt_cat} | Relevant Pool: {total_gt} reviews")

        # 1. Query understanding rewrite
        qu_res = query_understanding.analyze(q_text)
        search_query = qu_res.get("retrieval_query", q_text)

        # 2. Production retrieval (Dense + BM25 + RRF + BGE Reranker + Evidence Gate)
        res = production_retriever.retrieve(
            query=search_query,
            dense_top_k=50,
            bm25_top_k=50,
            final_top_k=k
        )
        retrieved_items = res.get("results", [])
        diagnostics = res.get("diagnostics", {})

        retrieved_ids = [item.get("review_id") for item in retrieved_items]
        
        # Calculate relevance
        relevant_retrieved = 0
        first_rel_rank = None

        for rank, r_id in enumerate(retrieved_ids, 1):
            is_rel = r_id in gt_ids
            rev_data = REVIEW_MAP.get(r_id, {})
            if is_rel:
                relevant_retrieved += 1
                if first_rel_rank is None:
                    first_rel_rank = rank
            
            status_mark = "✓ RELEVANT" if is_rel else "✗ OTHER"
            snippet = (rev_data.get("text", "")).replace("\n", " ")[:70]
            print(f"    Rank {rank}: [{status_mark}] Cat: {rev_data.get('category')} | Score: {retrieved_items[rank-1].get('reranker_score', 0):.4f} | \"{snippet}...\"")

        p_at_k = relevant_retrieved / max(1, len(retrieved_items)) if retrieved_items else 0.0
        r_at_k = relevant_retrieved / max(1, total_gt) if total_gt > 0 else 0.0
        rr = (1.0 / first_rel_rank) if first_rel_rank else 0.0
        hit = 1.0 if relevant_retrieved > 0 else 0.0

        overall_metrics[f"precision@{k}"].append(p_at_k)
        overall_metrics[f"recall@{k}"].append(r_at_k)
        overall_metrics["reciprocal_rank"].append(rr)
        overall_metrics["hit_rate"].append(hit)

        per_query_results.append({
            "query_id": q_id,
            "query": q_text,
            "target_category": tgt_cat,
            "retrieved_count": len(retrieved_items),
            "relevant_count": relevant_retrieved,
            f"precision@{k}": round(p_at_k, 4),
            f"recall@{k}": round(r_at_k, 4),
            "reciprocal_rank": round(rr, 4),
            "first_rank": first_rel_rank,
            "evidence_gate_status": diagnostics.get("evidence_gate_status")
        })

    # Compute macro-averages
    mean_p = sum(overall_metrics[f"precision@{k}"]) / len(BENCHMARK_CASES)
    mean_r = sum(overall_metrics[f"recall@{k}"]) / len(BENCHMARK_CASES)
    mrr = sum(overall_metrics["reciprocal_rank"]) / len(BENCHMARK_CASES)
    mean_hit = sum(overall_metrics["hit_rate"]) / len(BENCHMARK_CASES)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 80)
    print(f"{'Query ID':<16} | {'Target Category':<22} | {'Retrieved':<9} | {'Rel@k':<6} | {'P@k':<7} | {'RR':<6} | {'Gate'}")
    print("-" * 80)
    for row in per_query_results:
        print(f"{row['query_id']:<16} | {row['target_category']:<22} | {row['retrieved_count']:<9} | {row['relevant_count']:<6} | {row[f'precision@{k}']:.2f}   | {row['reciprocal_rank']:.2f}  | {row['evidence_gate_status']}")
    print("-" * 80)
    print(f"Macro-Average Precision@{k}: {mean_p:.4f} ({mean_p*100:.1f}%)")
    print(f"Macro-Average Recall@{k}:    {mean_r:.4f} ({mean_r*100:.1f}%)")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Hit Rate@{k}:               {mean_hit:.4f} ({mean_hit*100:.1f}%)")
    print("=" * 80)

    return {
        f"macro_precision@{k}": round(mean_p, 4),
        f"macro_recall@{k}": round(mean_r, 4),
        "mrr": round(mrr, 4),
        "hit_rate": round(mean_hit, 4),
        "queries": per_query_results
    }

if __name__ == "__main__":
    run_benchmark(k=8)
