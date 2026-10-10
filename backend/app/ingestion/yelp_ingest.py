import os
import sys
import json
import time
import uuid
import pickle
import argparse
from typing import Dict, Any, List, Set, Optional

# Offline environment flags for local HuggingFace cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import psycopg2
import torch
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.retrieval.hybrid_search import STOP_WORDS

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "archive")
STATE_FILE = os.path.join(DATA_DIR, "ingestion_state.json")
BM25_INDEX_FILE = os.path.join(DATA_DIR, "bm25_index.pkl")

# Qdrant collection constants
COLLECTION_NAME = "yelp_reviews_bge_m3"
VECTOR_DIM = 1024

def get_pg_connection(host: str, port: int, dbname: str, user: str, password: str = ""):
    """Connects to PostgreSQL database with fallback to current OS user."""
    try:
        return psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password if password else None
        )
    except psycopg2.Error as e:
        # Fallback to current OS user if specified user failed
        current_user = os.environ.get("USER", "bharathyadav")
        if user != current_user:
            return psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=current_user
            )
        raise e

def init_postgres_schema(conn):
    """Creates the production schema for Yelp businesses and reviews."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS yelp_businesses (
                business_id VARCHAR(64) PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT,
                state VARCHAR(16),
                stars REAL,
                review_count INT,
                categories TEXT
            );

            CREATE TABLE IF NOT EXISTS yelp_reviews (
                review_id VARCHAR(64) PRIMARY KEY,
                user_id VARCHAR(64),
                business_id VARCHAR(64),
                stars REAL,
                useful INT DEFAULT 0,
                funny INT DEFAULT 0,
                cool INT DEFAULT 0,
                text TEXT NOT NULL,
                date TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_yelp_reviews_biz ON yelp_reviews(business_id);
            CREATE INDEX IF NOT EXISTS idx_yelp_reviews_date ON yelp_reviews(date);
            CREATE INDEX IF NOT EXISTS idx_yelp_reviews_stars ON yelp_reviews(stars);
        """)
    conn.commit()

def init_qdrant_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME):
    """Initializes dedicated Qdrant collection with 1024-dimension Cosine distance."""
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if not exists:
        print(f"Creating Qdrant collection '{collection_name}' with dimension {VECTOR_DIM} (Cosine)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_DIM,
                distance=models.Distance.COSINE
            )
        )
    else:
        info = client.get_collection(collection_name)
        dim = info.config.params.vectors.size
        if dim != VECTOR_DIM:
            raise ValueError(
                f"Existing Qdrant collection '{collection_name}' has dimension {dim}, but {VECTOR_DIM} is required for BGE-M3!"
            )
        print(f"Verified Qdrant collection '{collection_name}' (dimension={dim}, vectors_count={info.points_count})")

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text removing punctuation and standard stopwords."""
    import re
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered = [t for t in tokens if t not in STOP_WORDS]
    return filtered if filtered else tokens

def ingest_businesses(businesses_path: str, conn) -> Dict[str, str]:
    """Streams and ingests Yelp businesses line-by-line; returns business_id -> name map."""
    if not os.path.exists(businesses_path):
        print(f"Warning: Businesses file not found at {businesses_path}. Skipping business metadata.")
        return {}

    print(f"\n--- Ingesting Yelp Businesses from {businesses_path} ---")
    biz_map = {}
    batch = []
    total_biz = 0
    start_t = time.time()

    with open(businesses_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            b = json.loads(line)
            b_id = b.get("business_id", "")
            b_name = b.get("name", "Unknown Business")
            biz_map[b_id] = b_name

            batch.append((
                b_id,
                b_name,
                b.get("city", ""),
                b.get("state", ""),
                float(b.get("stars", 0.0)),
                int(b.get("review_count", 0)),
                b.get("categories", "")
            ))

            if len(batch) >= 1000:
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO yelp_businesses (business_id, name, city, state, stars, review_count, categories)
                        VALUES %s
                        ON CONFLICT (business_id) DO NOTHING;
                        """,
                        batch
                    )
                conn.commit()
                total_biz += len(batch)
                batch = []

    if batch:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO yelp_businesses (business_id, name, city, state, stars, review_count, categories)
                VALUES %s
                ON CONFLICT (business_id) DO NOTHING;
                """,
                batch
            )
        conn.commit()
        total_biz += len(batch)

    print(f"Businesses Ingestion Complete: {total_biz} records processed into PostgreSQL in {time.time() - start_t:.2f}s")
    return biz_map

def run_ingestion(
    reviews_file: str,
    businesses_file: str,
    limit: int = 10000,
    batch_size: int = 500,
    emb_batch_size: int = 64,
    pg_host: str = "127.0.0.1",
    pg_port: int = 5432,
    pg_db: str = "postgres",
    pg_user: str = "bharathyadav",
    pg_pass: str = "",
    qdrant_url: str = "http://127.0.0.1:6333",
    collection_name: str = COLLECTION_NAME,
    resume: bool = True
):
    """
    Executes streaming batch ingestion of the real Yelp dataset into:
    1. PostgreSQL (Metadata & full review text)
    2. Qdrant (BGE-M3 1024-dim dense vectors)
    3. BM25 (Persistent lexical inverted index)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    start_total_time = time.time()

    print("==================================================")
    print("RESTAURANTOPS AI: PRODUCTION YELP INGESTION PIPELINE")
    print("==================================================")
    print(f"Source Review File:  {reviews_file}")
    print(f"Target Limit:        {limit if limit > 0 else 'FULL CORPUS (All records)'}")
    print(f"PostgreSQL Target:   {pg_host}:{pg_port}/{pg_db}")
    print(f"Qdrant Target:       {qdrant_url} (collection: {collection_name})")
    print(f"Embedding Model:     BAAI/bge-m3 (1024 dimensions)")
    print(f"Resumable:           {resume}")
    print("==================================================\n")

    # Step 1: Verify PostgreSQL
    print("[Phase 3] Connecting to PostgreSQL...")
    conn = get_pg_connection(pg_host, pg_port, pg_db, pg_user, pg_pass)
    init_postgres_schema(conn)
    print("PostgreSQL schema initialized successfully.\n")

    # Step 2: Verify Qdrant
    print("[Phase 4] Connecting to Qdrant...")
    qdrant = QdrantClient(url=qdrant_url)
    init_qdrant_collection(qdrant, collection_name)
    print("Qdrant vector collection verified.\n")

    # Step 3: Load BGE-M3 model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[Phase 5] Loading BAAI/bge-m3 embedding model on device={device}...", flush=True)
    embedder = SentenceTransformer("BAAI/bge-m3", device=device)
    actual_dim = embedder.get_sentence_embedding_dimension()
    if actual_dim != VECTOR_DIM:
        raise ValueError(f"BGE-M3 dimension mismatch! Expected {VECTOR_DIM}, got {actual_dim}")
    print(f"BGE-M3 loaded successfully! Vector Dimension = {actual_dim}\n", flush=True)

    # Step 4: Ingest Businesses
    biz_map = ingest_businesses(businesses_file, conn)

    # Step 5: Check existing records for Resumability
    existing_ids: Set[str] = set()
    if resume:
        print("[Resumability] Fetching already ingested review IDs from PostgreSQL...")
        with conn.cursor() as cur:
            cur.execute("SELECT review_id FROM yelp_reviews;")
            rows = cur.fetchall()
            existing_ids = {r[0] for r in rows}
        print(f"[Resumability] Found {len(existing_ids)} existing review IDs. These will be skipped.\n")

    # Step 6: Stream Reviews line-by-line
    print(f"[Streaming Ingestion] Starting review processing from {reviews_file}...")
    
    current_batch_reviews = []
    processed_count = 0
    inserted_count = 0
    failed_count = 0
    skipped_count = 0
    batch_idx = 0

    corpus_tokens = []
    bm25_review_ids = []

    # If persistent BM25 exists and resuming, load it
    if resume and os.path.exists(BM25_INDEX_FILE):
        try:
            with open(BM25_INDEX_FILE, "rb") as f:
                saved_bm25_data = pickle.load(f)
                corpus_tokens = saved_bm25_data.get("corpus_tokens", [])
                bm25_review_ids = saved_bm25_data.get("review_ids", [])
            print(f"[BM25] Loaded existing BM25 cache with {len(bm25_review_ids)} indexed documents.")
        except Exception as e:
            print(f"[BM25] Could not load prior cache: {e}. Starting fresh.")
            corpus_tokens = []
            bm25_review_ids = []

    t_loop_start = time.time()

    def flush_batch(batch):
        nonlocal inserted_count, failed_count
        if not batch:
            return

        # 1. PostgreSQL Batch Insert
        pg_records = [
            (
                r["review_id"],
                r.get("user_id", ""),
                r.get("business_id", ""),
                float(r.get("stars", 1.0)),
                int(r.get("useful", 0)),
                int(r.get("funny", 0)),
                int(r.get("cool", 0)),
                r.get("text", ""),
                r.get("date", "")
            )
            for r in batch
        ]
        try:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO yelp_reviews (review_id, user_id, business_id, stars, useful, funny, cool, text, date)
                    VALUES %s
                    ON CONFLICT (review_id) DO NOTHING;
                    """,
                    pg_records
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error inserting PostgreSQL batch: {e}")
            failed_count += len(batch)
            return

        # 2. BGE-M3 Embeddings Generation
        texts = [r.get("text", "") for r in batch]
        embeddings = embedder.encode(
            texts,
            batch_size=emb_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # 3. Qdrant Batch Upsert
        points = []
        for r, emb in zip(batch, embeddings):
            r_id = r["review_id"]
            # Convert review_id string to a valid UUIDv5 for Qdrant
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, r_id))
            b_id = r.get("business_id", "")
            b_name = biz_map.get(b_id, "Unknown Restaurant")
            payload = {
                "review_id": r_id,
                "business_id": b_id,
                "business_name": b_name,
                "stars": float(r.get("stars", 1.0)),
                "date": str(r.get("date", "")),
                "text": r.get("text", "")[:400] # Representative excerpt
            }
            points.append(
                models.PointStruct(
                    id=point_uuid,
                    vector=[float(x) for x in emb],
                    payload=payload
                )
            )

        # Chunked upsert to ensure robust HTTP payload sizing
        chunk_size = 100
        for i in range(0, len(points), chunk_size):
            qdrant.upsert(
                collection_name=collection_name,
                points=points[i:i + chunk_size]
            )

        # 4. BM25 Corpus Accumulation
        for r in batch:
            tokens = tokenize_for_bm25(r.get("text", ""))
            corpus_tokens.append(tokens)
            bm25_review_ids.append(r["review_id"])

        inserted_count += len(batch)

    with open(reviews_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            processed_count += 1
            r = json.loads(line)
            r_id = r.get("review_id", "")

            if resume and r_id in existing_ids:
                skipped_count += 1
                if limit > 0 and (inserted_count + skipped_count) >= limit:
                    break
                continue

            current_batch_reviews.append(r)

            if len(current_batch_reviews) >= batch_size:
                batch_idx += 1
                flush_batch(current_batch_reviews)
                current_batch_reviews = []

                elapsed = time.time() - t_loop_start
                rate = (inserted_count + skipped_count) / max(0.1, elapsed)
                print(
                    f"[Batch {batch_idx:04d}] Processed: {processed_count:,} | "
                    f"Inserted: {inserted_count:,} | Skipped: {skipped_count:,} | "
                    f"Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} rev/s",
                    flush=True
                )

            if limit > 0 and (inserted_count + skipped_count) >= limit:
                print(f"\n[Limit Reached] Target limit of {limit:,} reviews reached.")
                break

    # Flush final remaining batch
    if current_batch_reviews:
        batch_idx += 1
        flush_batch(current_batch_reviews)
        current_batch_reviews = []

    # Step 7: Build & Persist BM25 Index
    print("\n[Phase 6] Building persistent BM25 index over full ingested corpus...")
    t_bm25_start = time.time()
    bm25_model = BM25Okapi(corpus_tokens)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(
            {
                "corpus_tokens": corpus_tokens,
                "review_ids": bm25_review_ids,
                "bm25": bm25_model,
                "total_docs": len(bm25_review_ids)
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )
    print(f"BM25 index built and saved to {BM25_INDEX_FILE} ({len(bm25_review_ids):,} docs) in {time.time() - t_bm25_start:.2f}s")

    # Step 8: Save State
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "last_run": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_processed": processed_count,
                "total_inserted": inserted_count,
                "total_skipped": skipped_count,
                "total_failed": failed_count,
                "bm25_total_docs": len(bm25_review_ids),
                "collection_name": collection_name,
                "vector_dimension": VECTOR_DIM
            },
            f,
            indent=2
        )

    # Verification Counts
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM yelp_businesses;")
        pg_biz_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM yelp_reviews;")
        pg_rev_count = cur.fetchone()[0]

    qdrant_points = qdrant.get_collection(collection_name).points_count
    total_elapsed = time.time() - start_total_time

    print("\n==================================================")
    print("INGESTION EXECUTION SUMMARY")
    print("==================================================")
    print(f"Total Reviews Read from Source:    {processed_count:,}")
    print(f"Reviews Inserted in this Run:      {inserted_count:,}")
    print(f"Reviews Skipped (Already Ingested): {skipped_count:,}")
    print(f"Failed Records:                    {failed_count}")
    print(f"PostgreSQL Total Reviews:          {pg_rev_count:,}")
    print(f"PostgreSQL Total Businesses:       {pg_biz_count:,}")
    print(f"Qdrant Total Vectors (points):     {qdrant_points:,}")
    print(f"Qdrant Vector Dimension:           {VECTOR_DIM}")
    print(f"BM25 Indexed Document Count:       {len(bm25_review_ids):,}")
    print(f"Total Execution Time:              {total_elapsed:.2f}s")
    print(f"Effective Processing Rate:         {inserted_count / max(0.1, total_elapsed):.1f} rev/s")
    print("==================================================\n")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RestaurantOps AI: Offline Production Yelp Ingestion Pipeline")
    parser.add_argument("--limit", type=int, default=10000, help="Number of reviews to ingest (0 for full 7M corpus)")
    parser.add_argument("--batch-size", type=int, default=500, help="Database and BM25 batch size")
    parser.add_argument("--emb-batch-size", type=int, default=64, help="BGE-M3 embedding batch size")
    parser.add_argument("--reviews-file", type=str, default=os.path.join(ARCHIVE_DIR, "yelp_academic_dataset_review.json"))
    parser.add_argument("--businesses-file", type=str, default=os.path.join(ARCHIVE_DIR, "yelp_academic_dataset_business.json"))
    parser.add_argument("--pg-host", type=str, default="127.0.0.1")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-db", type=str, default="postgres")
    parser.add_argument("--pg-user", type=str, default=os.environ.get("USER", "bharathyadav"))
    parser.add_argument("--pg-pass", type=str, default="")
    parser.add_argument("--qdrant-url", type=str, default="http://127.0.0.1:6333")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME)
    parser.add_argument("--no-resume", action="store_true", help="Disable resumability and start fresh")

    args = parser.parse_args()

    run_ingestion(
        reviews_file=args.reviews_file,
        businesses_file=args.businesses_file,
        limit=args.limit,
        batch_size=args.batch_size,
        emb_batch_size=args.emb_batch_size,
        pg_host=args.pg_host,
        pg_port=args.pg_port,
        pg_db=args.pg_db,
        pg_user=args.pg_user,
        pg_pass=args.pg_pass,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        resume=not args.no_resume
    )
