import os
import sys
import json
import uuid
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REVIEWS_JSON = os.path.join(DATA_DIR, "yelp_reviews.json")
BIZ_JSON = os.path.join(DATA_DIR, "yelp_businesses.json")
COLLECTION_NAME = "yelp_reviews_bge_m3"
VECTOR_DIM = 1024

def sync():
    print("Loading 1,000 reviews and businesses from JSON...")
    with open(REVIEWS_JSON, "r", encoding="utf-8") as f:
        reviews = json.load(f)
    with open(BIZ_JSON, "r", encoding="utf-8") as f:
        businesses = json.load(f)

    print(f"Loaded {len(reviews)} reviews and {len(businesses)} businesses.")

    # 1. Sync PostgreSQL
    try:
        import psycopg2
        from psycopg2.extras import execute_values
        print("Connecting to PostgreSQL...")
        conn = psycopg2.connect(host="127.0.0.1", port=5432, dbname="postgres", user="bharathyadav", password="")
        with conn.cursor() as cur:
            # Upsert businesses
            biz_rows = [
                (b["business_id"], b.get("name"), b.get("city"), b.get("state"), float(b.get("stars", 0)), int(b.get("review_count", 0)), b.get("categories"))
                for b in businesses
            ]
            execute_values(
                cur,
                """
                INSERT INTO yelp_businesses (business_id, name, city, state, stars, review_count, categories)
                VALUES %s ON CONFLICT (business_id) DO NOTHING;
                """,
                biz_rows
            )
            # Upsert reviews
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
        print("Successfully synced 1,000 reviews into PostgreSQL!")
    except Exception as e:
        print(f"PostgreSQL sync notice: {e}")

    # 2. Sync Qdrant
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        from sentence_transformers import SentenceTransformer

        print("Connecting to Qdrant at http://127.0.0.1:6333...")
        qdrant = QdrantClient(url="http://127.0.0.1:6333")
        cols = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION_NAME not in cols:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE)
            )
            print(f"Created Qdrant collection '{COLLECTION_NAME}'.")

        print("Loading BGE-M3 model...")
        embedder = SentenceTransformer("BAAI/bge-m3")

        batch_size = 64
        total_upserted = 0
        t0 = time.time()

        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            texts = [r["text"] for r in batch]
            vecs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

            points = []
            for j, r in enumerate(batch):
                pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, r["review_id"]))
                points.append(
                    models.PointStruct(
                        id=pid,
                        vector=vecs[j].tolist(),
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

            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            total_upserted += len(points)
            print(f"Upserted {total_upserted}/{len(reviews)} vectors to Qdrant ({time.time() - t0:.1f}s)...")

        pts = qdrant.get_collection(COLLECTION_NAME).points_count
        print(f"Qdrant sync complete! Total collection points count: {pts}")
    except Exception as e:
        print(f"Qdrant sync notice: {e}")

if __name__ == "__main__":
    sync()
