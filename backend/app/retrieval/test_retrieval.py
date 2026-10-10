import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.retrieval.production_retriever import production_retriever

TEST_QUERIES = [
    "What are the biggest food quality complaints?",
    "What are customers saying about staff behavior?",
    "What are customers saying about long waiting times?",
    "What are the main pricing complaints?",
    "Did any customers report getting sick after eating?"
]

print("==================================================")
print("TESTING PRODUCTION RETRIEVAL ON REAL YELP DATA")
print("==================================================")

for q in TEST_QUERIES:
    print(f"\nQUERY: \"{q}\"", flush=True)
    res = production_retriever.retrieve(q, dense_top_k=50, bm25_top_k=50, final_top_k=8)
    diag = res["diagnostics"]
    results = res["results"]
    
    print(f"Diagnostics: Dense={diag['dense_count']}, BM25={diag['bm25_count']}, Fused={diag['fused_count']}, "
          f"Reranked={diag['reranked_count']}, Final={diag['final_count']}", flush=True)
    print(f"Evidence Gate: {diag['evidence_gate_status']} ({diag['evidence_gate_notes']})", flush=True)
    
    if not results:
        print("  -> [NO REVIEWS RETRIEVED - INSUFFICIENT EVIDENCE]", flush=True)
    else:
        for idx, r in enumerate(results, 1):
            excerpt = r['text'].replace('\n', ' ')[:120]
            print(f"  {idx}. [{r['business']}] (★{r['stars']}) Rerank={r.get('reranker_score', 0):.4f} | \"{excerpt}...\"", flush=True)
