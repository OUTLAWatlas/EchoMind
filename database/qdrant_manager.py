"""Qdrant manager for EchoMind.

Handles collection bootstrap with named vectors, cosine distance, scalar quantization,
payload indexing, and reinforcement upserts.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


class EchoMindDB:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        collection_name: str = "echomind_signs",
        hand_dim: int = 256,
        face_dim: int = 128,
    ) -> None:
        self.collection_name = collection_name
        self.hand_dim = hand_dim
        self.face_dim = face_dim
        self.client = QdrantClient(
            host=host or os.getenv("QDRANT_HOST", "localhost"),
            port=port or int(os.getenv("QDRANT_PORT", "6333")),
            api_key=api_key or os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True,
        )
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
            vectors=named_vectors,
            quantization_config=rest.ScalarQuantization(
                scalar=rest.ScalarQuantizationConfig(
                    type=rest.QuantizationType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
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

    def delete_point(self, point_id: str) -> None:
        """Delete a point by ID for undo/cleanup flows."""
        if not point_id:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=rest.PointIdsList(points=[point_id]),
            wait=True,
        )
