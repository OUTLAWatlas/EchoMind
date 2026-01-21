#!/usr/bin/env python3
"""Seed Qdrant Cloud with synthetic sign language gesture data.

High-performance ingestion script that:
- Connects to Qdrant Cloud using environment credentials
- Creates/validates echomind_signs collection with optimizations
- Generates realistic 256-dim vectors for essential signs
- Batches uploads with progress tracking
- Verifies ingestion with sample search

Usage:
    python seed_qdrant.py

Requirements:
    - QDRANT_URL and QDRANT_API_KEY in .env file
    - Dependencies: qdrant-client, numpy, python-dotenv, tqdm

Output:
    - 60 synthetic gesture points (20 signs × 3 variations)
    - Each point has gloss, dialect (ASL/ISL), success_count=0
    - Post-seed verification with sample search
"""
from __future__ import annotations

import os
import random
import uuid
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Collection configuration
COLLECTION_NAME = "echomind_signs"
VECTOR_DIM = 256  # Match existing collection schema
DISTANCE_METRIC = rest.Distance.COSINE

# Essential sign vocabulary (20 core signs)
ESSENTIAL_SIGNS = [
    "Hello",
    "Goodbye",
    "Yes",
    "No",
    "Please",
    "Thank You",
    "Sorry",
    "Help",
    "Stop",
    "Go",
    "Come",
    "Wait",
    "Food",
    "Water",
    "Doctor",
    "Emergency",
    "Family",
    "Friend",
    "Love",
    "Safe",
]

# Dialects for random assignment
DIALECTS = ["ASL", "ISL"]


def connect_to_qdrant() -> QdrantClient:
    """Connect to Qdrant Cloud using environment credentials.
    
    Returns:
        Configured QdrantClient instance.
        
    Raises:
        ValueError: If credentials are missing.
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url or not api_key:
        raise ValueError(
            "Missing Qdrant credentials. Ensure QDRANT_URL and QDRANT_API_KEY "
            "are set in your .env file."
        )
    
    # Ensure URL includes port
    if ":6333" not in url and not url.endswith(":6333"):
        url = url.rstrip("/") + ":6333"
    
    client = QdrantClient(url=url, api_key=api_key, timeout=30)
    print(f"✓ Connected to Qdrant Cloud: {url}")
    return client


def ensure_collection(client: QdrantClient) -> None:
    """Create collection if it doesn't exist, with performance optimizations.
    
    Args:
        client: Connected QdrantClient instance.
    """
    if client.collection_exists(COLLECTION_NAME):
        print(f"✓ Collection '{COLLECTION_NAME}' already exists")
        # Update collection config for optimizations if needed
        try:
            client.update_collection(
                collection_name=COLLECTION_NAME,
                optimizer_config=rest.OptimizersConfigDiff(
                    indexing_threshold=10000,  # Start indexing after 10k points
                ),
            )
            print("  → Updated optimizer config")
        except Exception as e:
            print(f"  → Optimizer update skipped: {e}")
        return
    
    print(f"Creating collection '{COLLECTION_NAME}' with optimizations...")
    
    # Create collection with named vectors (matching existing schema)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "hand_motion": rest.VectorParams(
                size=VECTOR_DIM,
                distance=DISTANCE_METRIC,
                on_disk=True,  # Enable on-disk storage for scalability
                hnsw_config=rest.HnswConfigDiff(
                    m=32,
                    ef_construct=128,
                    on_disk=True,  # HNSW index on disk
                ),
                quantization_config=rest.ScalarQuantization(
                    scalar=rest.ScalarQuantizationConfig(
                        type=rest.ScalarQuantizationType.INT8,
                        quantile=0.99,
                        always_ram=False,  # Allow disk storage
                    )
                ),
            )
        },
        optimizers_config=rest.OptimizersConfigDiff(
            indexing_threshold=10000,
        ),
    )
    
    # Create payload indexes for fast filtering
    print("  → Creating payload indexes...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="gloss",
        field_schema=rest.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="dialect",
        field_schema=rest.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="success_count",
        field_schema=rest.PayloadSchemaType.INTEGER,
    )
    
    print(f"✓ Collection '{COLLECTION_NAME}' created with on-disk + scalar quantization")


def generate_realistic_vector(seed: str) -> List[float]:
    """Generate a deterministic but realistic 256-dim gesture vector.
    
    Uses a seeded random generator to create consistent vectors for the same sign,
    simulating hand position/velocity features in normalized space.
    
    Args:
        seed: Seed string (e.g., sign gloss) for reproducibility.
        
    Returns:
        256-dimensional normalized vector.
    """
    # Seed numpy RNG for reproducibility
    rng = np.random.RandomState(hash(seed) % (2**32))
    
    # Generate sparse vector with some structure
    # Simulate: hand positions (128), velocities (64), orientations (64)
    hand_positions = rng.randn(128) * 0.3  # Left/right hand positions in 3D space over time
    velocities = rng.randn(64) * 0.5  # Movement deltas
    orientations = rng.randn(64) * 0.2  # Hand orientation features
    
    vector = np.concatenate([hand_positions, velocities, orientations])
    
    # L2 normalization for cosine similarity
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector.astype(np.float32).tolist()


def generate_seed_library() -> List[Dict]:
    """Generate synthetic data for essential signs.
    
    Returns:
        List of point dictionaries ready for batch upload.
    """
    points = []
    
    for sign in ESSENTIAL_SIGNS:
        # Generate multiple variations per sign (different performers, speeds)
        for variation in range(3):  # 3 variations per sign
            point_id = str(uuid.uuid4())
            
            # Add variation to seed for different vectors
            seed = f"{sign}_v{variation}"
            vector = generate_realistic_vector(seed)
            
            # Random dialect assignment
            dialect = random.choice(DIALECTS)
            
            point = {
                "id": point_id,
                "vector": {"hand_motion": vector},
                "payload": {
                    "gloss": sign,
                    "dialect": dialect,
                    "success_count": 0,
                    "variation": variation,
                    "source": "seed_script",
                },
            }
            points.append(point)
    
    return points


def batch_upload_points(client: QdrantClient, points: List[Dict], batch_size: int = 100) -> None:
    """Upload points in batches with progress tracking.
    
    Args:
        client: Connected QdrantClient instance.
        points: List of point dictionaries.
        batch_size: Number of points per batch.
    """
    total = len(points)
    print(f"\n📤 Uploading {total} points in batches of {batch_size}...")
    
    # Convert to PointStruct objects
    point_structs = []
    for p in points:
        point_structs.append(
            rest.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"],
            )
        )
    
    # Batch upload with progress bar
    with tqdm(total=total, desc="Ingesting", unit="pts") as pbar:
        for i in range(0, total, batch_size):
            batch = point_structs[i:i + batch_size]
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True,  # Ensure consistency
            )
            pbar.update(len(batch))
    
    print("✓ Upload complete!")


def verify_ingestion(client: QdrantClient) -> None:
    """Verify ingestion with collection stats and sample search.
    
    Args:
        client: Connected QdrantClient instance.
    """
    print("\n📊 Post-Seed Verification")
    print("=" * 50)
    
    # Get collection info
    info = client.get_collection(COLLECTION_NAME)
    point_count = info.points_count
    print(f"Total Points: {point_count:,}")
    print(f"Vector Config: {VECTOR_DIM}D, {DISTANCE_METRIC}")
    print(f"Indexed: {info.status}")
    
    # Sample search test
    print("\n🔍 Test Search (query: 'Hello' vector)")
    test_vector = generate_realistic_vector("Hello_v0")
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=test_vector,
        using="hand_motion",
        limit=5,
        with_payload=True,
    )
    
    if hasattr(results, 'points'):
        points = results.points
    else:
        points = results
    
    if points:
        print(f"\nTop {len(points)} Results:")
        for i, hit in enumerate(points, 1):
            score = getattr(hit, 'score', 'N/A')
            payload = getattr(hit, 'payload', {})
            gloss = payload.get('gloss', 'Unknown')
            dialect = payload.get('dialect', 'N/A')
            print(f"  {i}. {gloss} ({dialect}) - Score: {score:.4f}")
    else:
        print("  No results found (collection may be empty)")
    
    print("\n" + "=" * 50)
    print("✅ Seeding complete! Your Qdrant Cloud cluster is warmed up.")


def main():
    """Main execution flow."""
    print("🚀 EchoMind Qdrant Cloud Seeding Script")
    print("=" * 50)
    
    # Step 1: Connect
    client = connect_to_qdrant()
    
    # Step 2: Ensure collection exists with optimizations
    ensure_collection(client)
    
    # Step 3: Generate seed library
    print(f"\n🌱 Generating seed library ({len(ESSENTIAL_SIGNS)} signs × 3 variations)...")
    seed_points = generate_seed_library()
    print(f"✓ Generated {len(seed_points)} synthetic gesture points")
    
    # Step 4: High-performance batched upload
    batch_upload_points(client, seed_points, batch_size=50)
    
    # Step 5: Verify ingestion
    verify_ingestion(client)


if __name__ == "__main__":
    main()
