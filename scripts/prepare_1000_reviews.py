import os
import sys
import json
import time
import uuid
import pickle
import numpy as np

# Force offline mode for local HF cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.retrieval.hybrid_search import STOP_WORDS

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "archive")
BIZ_FILE = os.path.join(ARCHIVE_DIR, "yelp_academic_dataset_business.json")
REV_FILE = os.path.join(ARCHIVE_DIR, "yelp_academic_dataset_review.json")

TARGET_REVIEWS_JSON = os.path.join(DATA_DIR, "yelp_reviews.json")
TARGET_BIZ_JSON = os.path.join(DATA_DIR, "yelp_businesses.json")
BM25_INDEX_FILE = os.path.join(DATA_DIR, "bm25_index.pkl")

QDRANT_COLLECTION = "yelp_reviews_bge_m3"
VECTOR_DIM = 1024
TARGET_COUNT = 1000

def tokenize_for_bm25(text: str):
    import re
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered = [t for t in tokens if t not in STOP_WORDS]
    return filtered if filtered else tokens

def run():
    print(f"==================================================")
    print(f"INGESTING {TARGET_COUNT:,} REAL YELP REVIEWS FROM ARCHIVE")
    print(f"==================================================")
    t0 = time.time()

    # Step 1: Scan restaurant businesses
    print("\n[1/5] Scanning restaurant businesses from archive...")
    restaurant_biz = {}
    with open(BIZ_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            b = json.loads(line)
            cats = (b.get("categories") or "").lower()
            if any(term in cats for term in ["restaurant", "food", "bar", "cafe", "bakery", "diner", "pizza", "burger", "bistro", "pub", "steak"]):
                restaurant_biz[b["business_id"]] = b

    print(f"Loaded {len(restaurant_biz):,} restaurant profiles.")

    # Step 2: Stream 1,000 reviews
    print(f"\n[2/5] Streaming {TARGET_COUNT:,} real reviews from {REV_FILE}...")
    reviews = []
    used_biz_ids = set()

    with open(REV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            b_id = r.get("business_id")
            if b_id in restaurant_biz:
                b = restaurant_biz[b_id]
                stars = float(r.get("stars", 1.0))
                text = r.get("text", "").strip()
                lower_t = text.lower()

                # Operational Aspect & Category assignment
                if any(w in lower_t for w in ["poison", "sick", "vomit", "nausea", "hair", "staple", "dirty", "filthy", "hygiene", "cockroach", "bug"]):
                    category = "Food Safety & Hygiene"
                    aspect = "Contamination & Health Risk"
                    urgency = "Critical"
                    severity = "Critical"
                elif any(w in lower_t for w in ["slow", "wait", "minute", "hour", "delay", "line", "forever", "late"]):
                    category = "Service Speed"
                    aspect = "Kitchen Ticket & Seating Pacing"
                    urgency = "High" if stars <= 2 else "Medium"
                    severity = "High" if stars <= 2 else "Low"
                elif any(w in lower_t for w in ["cold", "undercooked", "raw", "burnt", "salty", "bland", "tasteless", "flavor", "inedible", "taste"]):
                    category = "Food Quality"
                    aspect = "Recipe Execution & Thermal Control"
                    urgency = "High" if stars <= 2 else "Low"
                    severity = "High" if stars <= 2 else "Low"
                elif any(w in lower_t for w in ["rude", "attitude", "ignore", "server", "waiter", "hostess", "disrespectful", "unfriendly"]):
                    category = "Staff Demeanor"
                    aspect = "Hospitality & Guest Interaction"
                    urgency = "Medium"
                    severity = "Medium"
                elif any(w in lower_t for w in ["expensive", "overpriced", "charge", "bill", "surcharge", "rip off"]):
                    category = "Pricing & Value"
                    aspect = "Transparent Billing & Fair Value"
                    urgency = "Medium"
                    severity = "Medium"
                else:
                    category = "General Operations"
                    aspect = "Dining Experience & Ambiance"
                    urgency = "Low"
                    severity = "Low"

                item = {
                    "review_id": r.get("review_id"),
                    "business_id": b_id,
                    "user_id": r.get("user_id"),
                    "stars": stars,
                    "date": str(r.get("date")),
                    "text": text,
                    "useful": int(r.get("useful", 0)),
                    "funny": int(r.get("funny", 0)),
                    "cool": int(r.get("cool", 0)),
                    "restaurant_name": b.get("name", "Unknown Restaurant"),
                    "city": b.get("city", ""),
                    "state": b.get("state", ""),
                    "category": category,
                    "aspect": aspect,
                    "urgency": urgency,
                    "severity": severity
                }
                reviews.append(item)
                used_biz_ids.add(b_id)

                if len(reviews) >= TARGET_COUNT:
                    break

    print(f"Extracted {len(reviews):,} real reviews spanning {len(used_biz_ids):,} restaurants.")

    # Save to data/yelp_reviews.json
    print(f"\n[3/5] Saving to {TARGET_REVIEWS_JSON} and {TARGET_BIZ_JSON}...")
    with open(TARGET_REVIEWS_JSON, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2)

    businesses_to_save = [restaurant_biz[bid] for bid in used_biz_ids if bid in restaurant_biz]
    with open(TARGET_BIZ_JSON, "w", encoding="utf-8") as f:
        json.dump(businesses_to_save, f, indent=2)

    print(f"Saved {len(reviews)} reviews and {len(businesses_to_save)} businesses to disk.")

    # Step 4: Build BM25 Index
    print(f"\n[4/5] Building BM25 index over {len(reviews):,} reviews...")
    from rank_bm25 import BM25Okapi
    corpus_tokens = []
    review_ids = []
    for r in reviews:
        tokens = tokenize_for_bm25(r["text"])
        corpus_tokens.append(tokens)
        review_ids.append(r["review_id"])

    bm25 = BM25Okapi(corpus_tokens)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(
            {
                "corpus_tokens": corpus_tokens,
                "review_ids": review_ids,
                "bm25": bm25,
                "total_docs": len(review_ids)
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )
    print(f"Saved BM25 index with {len(review_ids):,} docs to {BM25_INDEX_FILE}.")

    # Step 5: Encode vectors with BGE-M3 and upsert to Qdrant
    print(f"\n[5/5] Generating BGE-M3 1024-dim embeddings & upserting to Qdrant...")
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("BAAI/bge-m3")
    print("BGE-M3 embedder loaded.")

    try:
        qdrant = QdrantClient(url="http://127.0.0.1:6333")
        # Ensure collection exists with dimension 1024
        cols = [c.name for c in qdrant.get_collections().collections]
        if QDRANT_COLLECTION not in cols:
            qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE)
            )
            print(f"Created Qdrant collection '{QDRANT_COLLECTION}'.")

        # Encode in batches
        batch_size = 64
        total_upserted = 0
        texts = [r["text"] for r in reviews]

        for i in range(0, len(reviews), batch_size):
            batch_slice = reviews[i:i + batch_size]
            batch_texts = [r["text"] for r in batch_slice]
            vectors = embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)

            points = []
            for j, r in enumerate(batch_slice):
                # Deterministic UUID from review_id
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, r["review_id"]))
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vectors[j].tolist(),
                        payload={
                            "review_id": r["review_id"],
                            "business_id": r["business_id"],
                            "restaurant_name": r["restaurant_name"],
                            "city": r["city"],
                            "state": r["state"],
                            "stars": r["stars"],
                            "date": r["date"],
                            "text": r["text"],
                            "category": r["category"],
                            "aspect": r["aspect"],
                            "urgency": r["urgency"],
                            "severity": r["severity"]
                        }
                    )
                )

            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
            total_upserted += len(points)
            print(f"Upserted {total_upserted}/{len(reviews)} vectors into Qdrant...")

        print(f"Qdrant indexing complete! Total points: {qdrant.get_collection(QDRANT_COLLECTION).points_count}")
    except Exception as e:
        print(f"Notice: Qdrant upsert encountered: {e}")

    # Optional PostgreSQL Sync
    try:
        import psycopg2
        from psycopg2.extras import execute_values
        conn = psycopg2.connect(host="127.0.0.1", port=5432, dbname="postgres", user="bharathyadav", password="")
        with conn.cursor() as cur:
            biz_rows = [
                (b["business_id"], b.get("name"), b.get("city"), b.get("state"), float(b.get("stars", 0)), int(b.get("review_count", 0)), b.get("categories"))
                for b in businesses_to_save
            ]
            execute_values(
                cur,
                """
                INSERT INTO yelp_businesses (business_id, name, city, state, stars, review_count, categories)
                VALUES %s ON CONFLICT (business_id) DO NOTHING;
                """,
                biz_rows
            )
            rev_rows = [
                (r["review_id"], r.get("user_id"), r["business_id"], float(r.get("stars", 1)), int(r.get("useful", 0)), int(r.get("funny", 0)), int(r.get("cool", 0)), r["text"], r.get("date"))
                for r in reviews
            ]
            execute_values(
                cur,
                """
                INSERT INTO yelp_reviews (review_id, user_id, business_id, stars, useful, funny, cool, text, date)
                VALUES %s ON CONFLICT (review_id) DO NOTHING;
                """,
                rev_rows
            )
        conn.commit()
        conn.close()
        print("Synced with PostgreSQL.")
    except Exception as e:
        print(f"PostgreSQL note: {e}")

    print(f"\nSUCCESS: Pipeline complete in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    run()
