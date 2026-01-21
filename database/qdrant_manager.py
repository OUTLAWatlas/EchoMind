"""Qdrant manager for EchoMind.

Handles collection bootstrap with named vectors, cosine distance, scalar quantization,
payload indexing, and reinforcement upserts.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Load environment variables
load_dotenv()


class EchoMindDB:
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "echomind_signs",
        hand_dim: int = 256,
        face_dim: int = 128,
    ) -> None:
        self.collection_name = collection_name
        self.hand_dim = hand_dim
        self.face_dim = face_dim
        
        # Priority: Cloud URL > local host > in-memory
        use_memory = os.getenv("QDRANT_USE_MEMORY", "false").lower() == "true"
        cloud_url = url or os.getenv("QDRANT_URL") or os.getenv("Qdrant_URL")
        cloud_key = api_key or os.getenv("QDRANT_API_KEY")
        
        if use_memory:
            self.client = QdrantClient(":memory:")
            print("Using in-memory Qdrant database.")
        elif cloud_url:
            # Qdrant Cloud connection
            try:
                # Ensure URL includes port if not already present
                if ":6333" not in cloud_url and not cloud_url.endswith(":6333"):
                    cloud_url = cloud_url.rstrip("/") + ":6333"
                self.client = QdrantClient(
                    url=cloud_url,
                    api_key=cloud_key,
                    timeout=10,
                )
                # Test connection
                self.client.get_collections()
                print(f"Connected to Qdrant Cloud: {cloud_url}")
            except Exception as e:
                print(f"Warning: Could not connect to Qdrant Cloud ({e}). Using in-memory database.")
                self.client = QdrantClient(":memory:")
        else:
            # Local Qdrant server
            try:
                self.client = QdrantClient(
                    host=os.getenv("QDRANT_HOST", "localhost"),
                    port=int(os.getenv("QDRANT_PORT", "6333")),
                    api_key=cloud_key,
                    prefer_grpc=True,
                    timeout=5,
                )
                self.client.get_collections()
                print("Connected to local Qdrant server.")
            except Exception as e:
                print(f"Warning: Could not connect to Qdrant server ({e}). Using in-memory database.")
                self.client = QdrantClient(":memory:")
        
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        named_vectors = {
            "hand_motion": rest.VectorParams(
                size=self.hand_dim,
                distance=rest.Distance.COSINE,
                hnsw_config=rest.HnswConfigDiff(m=32, ef_construct=128),
            ),
            "face_expression": rest.VectorParams(
                size=self.face_dim,
                distance=rest.Distance.COSINE,
                hnsw_config=rest.HnswConfigDiff(m=32, ef_construct=128),
            ),
        }

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=named_vectors,
        )

        # Payload indexing for fast filters and counters.
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="gloss",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="dialect",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="success_count",
            field_schema=rest.PayloadSchemaType.INTEGER,
        )

    def upsert_correction(
        self,
        vectors: Dict[str, Any],
        payload: Dict[str, Any],
        point_id: Optional[str] = None,
        increment_success: bool = True,
    ) -> str:
        """Upsert corrected gesture to satisfy long-term memory.

        Args:
            vectors: Named vectors matching the collection schema.
            payload: Metadata including gloss/dialect/success_count.
            point_id: Optional specific ID; generated if absent.
            increment_success: If True, bumps success_count before upsert.
        Returns:
            The point ID written to Qdrant.
        """
        pid = point_id or str(uuid.uuid4())
        if increment_success:
            payload["success_count"] = int(payload.get("success_count", 0)) + 1

        point = rest.PointStruct(id=pid, vector=vectors, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point], wait=True)
        return pid

    def search_gesture(
        self,
        query_vector: List[float],
        dialect: str,
        vector_name: str = "hand_motion",
        limit: int = 5,
    ) -> List[rest.ScoredPoint]:
        """Search for similar gestures with dialect filtering.
        
        Args:
            query_vector: 128-dimensional or 256-dimensional query vector.
            dialect: Dialect filter (e.g., 'ASL', 'ISL').
            vector_name: Named vector to search (hand_motion or face_expression).
            limit: Number of results to return.
            
        Returns:
            List of ScoredPoint objects with cosine similarity scores.
        """
        try:
            print(f"[SEARCH] Querying {vector_name} with dialect={dialect}, limit={limit}, vector_len={len(query_vector)}")
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=vector_name,
                query_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="dialect",
                            match=rest.MatchValue(value=dialect),
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
            )
            points = results.points if hasattr(results, 'points') else results
            print(f"[SEARCH] Got {len(points)} results")
            return points
        except Exception as e:
            print(f"[SEARCH ERROR] {e}")
            import traceback
            traceback.print_exc()
            return []

    def delete_point(self, point_id: str) -> None:
        """Delete a point by ID for undo/cleanup flows."""
        if not point_id:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=rest.PointIdsList(points=[point_id]),
            wait=True,
        )
