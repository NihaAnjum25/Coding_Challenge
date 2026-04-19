from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
except ModuleNotFoundError:
    MongoClient = None  # type: ignore[assignment]
    UpdateOne = None  # type: ignore[assignment]
    Collection = Any  # type: ignore[assignment]
    PyMongoError = Exception  # type: ignore[assignment]
    ServerSelectionTimeoutError = Exception  # type: ignore[assignment]

try:
    import mongomock
except ModuleNotFoundError:
    mongomock = None  # type: ignore[assignment]


@dataclass
class DocumentItem:
    """Represents one document with metadata for storage."""

    doc_id: str
    content: str
    metadata: Dict[str, Any]


def load_documents() -> List[DocumentItem]:
    """Prepare sample documents for MongoDB storage."""
    return [
        DocumentItem(
            doc_id="doc_001",
            content="FAISS enables fast similarity search across embedding vectors.",
            metadata={"source": "ai_blog", "category": "vector_search", "tags": ["faiss", "retrieval"]},
        ),
        DocumentItem(
            doc_id="doc_002",
            content="RAG pipelines combine retrieval with language model generation.",
            metadata={"source": "ml_notes", "category": "rag", "tags": ["rag", "llm", "knowledge"]},
        ),
        DocumentItem(
            doc_id="doc_003",
            content="MongoDB stores flexible JSON-like documents for scalable applications.",
            metadata={"source": "db_guide", "category": "database", "tags": ["mongodb", "nosql"]},
        ),
        DocumentItem(
            doc_id="doc_004",
            content="Embedding metadata helps filter results by source, topic, or timestamp.",
            metadata={"source": "engineering_wiki", "category": "pipeline", "tags": ["metadata", "embeddings"]},
        ),
    ]


def _hash_token(token: str, dim: int) -> int:
    """Map a token to one stable dimension index."""
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


def build_embedding(text: str, dim: int = 64) -> List[float]:
    """
    Create a deterministic dense embedding using a hashing-based method.
    This keeps the challenge runnable without external model downloads.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = [t.lower().strip(".,!?;:()[]{}") for t in text.split() if t.strip()]
    for token in tokens:
        index = _hash_token(token, dim)
        vec[index] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.astype(np.float32).tolist()


def build_records(documents: List[DocumentItem], embedding_model: str = "hashing_v1", dim: int = 64) -> List[Dict[str, Any]]:
    """Build MongoDB-ready records with metadata and embeddings."""
    now = datetime.now(timezone.utc)
    records: List[Dict[str, Any]] = []

    for item in documents:
        records.append(
            {
                "_id": item.doc_id,
                "content": item.content,
                "metadata": item.metadata,
                "embedding": build_embedding(item.content, dim=dim),
                "embedding_model": embedding_model,
                "embedding_dim": dim,
                "created_at": now,
                "updated_at": now,
            }
        )
    return records


def connect_collection() -> tuple[Collection, str]:
    """Connect to MongoDB collection, with optional local fallback."""
    if MongoClient is None:
        raise ModuleNotFoundError("pymongo is not installed.")

    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "ai_challenge")
    collection_name = os.getenv("MONGODB_COLLECTION", "document_embeddings")

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
        return client[db_name][collection_name], "mongodb"
    except ServerSelectionTimeoutError:
        if mongomock is None:
            raise
        mock_client = mongomock.MongoClient()
        return mock_client[db_name][collection_name], "mongomock"


def create_indexes(collection: Collection) -> None:
    """Create indexes for efficient filtering and metadata lookups."""
    collection.create_index("metadata.category")
    collection.create_index("metadata.source")
    collection.create_index("updated_at")


def upsert_records(collection: Collection, records: List[Dict[str, Any]]) -> int:
    """Insert or update records with upsert (compatible with MongoDB and mongomock)."""
    affected = 0
    for record in records:
        result = collection.replace_one({"_id": record["_id"]}, record, upsert=True)
        if result.upserted_id is not None or result.modified_count > 0:
            affected += 1
    return affected


def show_sample_records(collection: Collection, limit: int = 3) -> None:
    """Display stored records with metadata and embedding stats."""
    print("\nStored documents (sample):")
    cursor = collection.find({}, {"content": 1, "metadata": 1, "embedding_dim": 1, "embedding": 1}).limit(limit)

    for item in cursor:
        print(f"\n_id: {item.get('_id')}")
        print(f"content: {item.get('content')}")
        print(f"metadata: {item.get('metadata')}")
        embedding = item.get("embedding", [])
        print(f"embedding_dim: {item.get('embedding_dim')}, stored_values: {len(embedding)}")


def main() -> None:
    """
    Day 40 Challenge: MongoDB document + embedding storage
    1. Prepare document content and metadata
    2. Generate embeddings for each document
    3. Store everything in MongoDB with schema-friendly fields
    4. Show sample stored records
    """
    documents = load_documents()
    records = build_records(documents, embedding_model="hashing_v1", dim=64)

    print("Day 40 - MongoDB Document + Embedding Storage")
    print(f"Prepared records: {len(records)}")

    try:
        collection, backend = connect_collection()
        create_indexes(collection)
        affected = upsert_records(collection, records)
        total = collection.count_documents({})

        print("MongoDB connection: SUCCESS")
        print(f"Storage backend: {backend}")
        print(f"Records inserted/updated: {affected}")
        print(f"Total records in collection: {total}")
        show_sample_records(collection, limit=3)

    except ModuleNotFoundError:
        print("MongoDB connection: SKIPPED")
        print("Reason: PyMongo is not installed.")
        print("Install with: pip install pymongo")
    except ServerSelectionTimeoutError:
        print("MongoDB connection: FAILED")
        print("Reason: Could not connect to MongoDB server.")
        print("Start MongoDB locally or set MONGODB_URI to your running instance.")
    except PyMongoError as error:
        print("MongoDB operation: FAILED")
        print(f"Reason: {error}")


if __name__ == "__main__":
    main()
