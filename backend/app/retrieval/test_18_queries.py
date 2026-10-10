import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from backend.app.query_understanding.query_analyzer import query_understanding
from backend.app.retrieval.production_retriever import production_retriever

TEST_QUERIES = [
    # WAIT TIME
    ("WAIT TIME", "What are the complaints about long waiting times?"),
    ("WAIT TIME", "I am having a delay in service"),
    ("WAIT TIME", "Service is very late"),
    ("WAIT TIME", "Orders are taking too long"),
    ("WAIT TIME", "Customers are waiting forever"),
    ("WAIT TIME", "Why is service so slow?"),
    
    # FOOD QUALITY
    ("FOOD QUALITY", "What are the biggest food quality complaints?"),
    ("FOOD QUALITY", "The food quality is bad"),
    ("FOOD QUALITY", "Customers are unhappy with the food"),
    
    # STAFF
    ("STAFF", "What are customers saying about staff behavior?"),
    ("STAFF", "The staff are rude"),
    ("STAFF", "Customers don't like how employees behave"),
    
    # PRICING
    ("PRICING", "What are the main pricing complaints?"),
    ("PRICING", "Customers think it is too expensive"),
    ("PRICING", "The prices are too high"),
    
    # ILLNESS
    ("ILLNESS", "Did any customers report getting sick?"),
    ("ILLNESS", "Did anyone become ill after eating?"),
    ("ILLNESS", "Any reports of food poisoning?")
]

print("=" * 80)
print("TESTING 18 QUERIES WITH SEMANTIC QUERY UNDERSTANDING LAYER")
print("=" * 80)

summary_table = []

for idx, (cat, orig_q) in enumerate(TEST_QUERIES, 1):
    print(f"\n[{idx}/18] Category: {cat}")
    print(f"Original query: \"{orig_q}\"")
    
    # 1. Query Understanding
    qu = query_understanding.analyze(orig_q)
    det_intent = qu["intent"]
    det_aspect = qu["aspect"]
    search_rep = qu["retrieval_query"]
    conf = qu["confidence"]
    
    print(f"  ↓ Detected intent: {det_intent}")
    print(f"  ↓ Detected aspect: {det_aspect} (confidence: {conf})")
    print(f"  ↓ Generated semantic search representation: \"{search_rep}\"")
    
    # 2. Production Multi-Stage Retrieval
    res = production_retriever.retrieve(
        query=search_rep,
        dense_top_k=50,
        bm25_top_k=50,
        final_top_k=8
    )
    diag = res["diagnostics"]
    results = res["results"]
    
    dense_cnt = diag["dense_count"]
    bm25_cnt = diag["bm25_count"]
    fused_cnt = diag["fused_count"]
    rerank_cnt = diag["reranked_count"]
    final_cnt = diag["final_count"]
    eg_status = diag["evidence_gate_status"]
    
    print(f"  ↓ Dense results: {dense_cnt}")
    print(f"  ↓ BM25 results: {bm25_cnt}")
    print(f"  ↓ RRF results (fused & deduped): {fused_cnt}")
    print(f"  ↓ Final reranked evidence: {final_cnt} (of {rerank_cnt} reranked)")
    print(f"  ↓ Evidence Gate result: {eg_status}")
    
    if results:
        for r_i, r in enumerate(results[:3], 1):
            text_snippet = r['text'].replace('\n', ' ')[:90]
            score_val = r.get('reranker_score', r.get('rrf_score', 0))
            print(f"     {r_i}. [{r['business']}] (★{r['stars']}, score: {score_val:.4f}): \"{text_snippet}...\"")
    else:
        print("     [NO REVIEWS RETRIEVED - INSUFFICIENT EVIDENCE]")
        
    summary_table.append({
        "num": idx,
        "category": cat,
        "query": orig_q,
        "aspect": det_aspect,
        "intent": det_intent,
        "final_count": final_cnt,
        "gate": eg_status
    })

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'#':<3} | {'Expected Cat':<13} | {'Detected Aspect':<22} | {'Intent':<20} | {'Retrieved':<9} | {'Gate Status'}")
print("-" * 80)
for row in summary_table:
    print(f"{row['num']:<3} | {row['category']:<13} | {str(row['aspect']):<22} | {row['intent']:<20} | {row['final_count']:<9} | {row['gate']}")
print("=" * 80)
