"""GestureController for real-time temporal buffer management and search.

Maintains a sliding window of normalized pose landmarks, coordinates SignEncoder,
handles swipe-based interrupts, and dispatches hybrid Qdrant searches on motion
boundaries without blocking the UI.
"""
from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional, Protocol

import numpy as np
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from core.gesture_processor import LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST
from database.qdrant_manager import EchoMindDB


class SignEncoder(Protocol):
    def encode_motion(self, landmarks_seq: Iterable[np.ndarray]) -> Optional[List[float]]:
        ...


class SwipeHandler(Protocol):
    def detect_undo_swipe(self, pose_landmarks: np.ndarray) -> bool:
        ...


@dataclass
class ControllerConfig:
    buffer_size: int = 45  # frames (~1.5s at 30 FPS)
    min_frames: int = 8
    velocity_start: float = 0.05
    neutral_offset: float = 0.15
    neutral_frames: int = 5
    search_top_k: int = 3


@dataclass
class GestureState:
    status: str = "Waiting"
    last_results: Optional[List] = None
    neutral_counter: int = 0
    last_vector: Optional[List[float]] = None
    last_payload: Optional[Dict] = None
    last_upsert_id: Optional[str] = None


class GestureController:
    def __init__(
        self,
        encoder: SignEncoder,
        swipe_handler: SwipeHandler,
        db: EchoMindDB,
        config: Optional[ControllerConfig] = None,
    ) -> None:
        self.cfg = config or ControllerConfig()
        self.encoder = encoder
        self.swipe_handler = swipe_handler
        self.db = db
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.cfg.buffer_size)
        self.state = "IDLE"
        self.prev_wrists: Optional[np.ndarray] = None
        self.gs = GestureState()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _torso_center_and_scale(self, pose: np.ndarray) -> tuple[np.ndarray, float]:
        shoulders = pose[[LEFT_SHOULDER, RIGHT_SHOULDER]]
        center = shoulders.mean(axis=0)
        scale = np.linalg.norm(shoulders[0, :2] - shoulders[1, :2]) + 1e-6
        return center, scale

    def _normalize(self, pose: np.ndarray) -> np.ndarray:
        center, scale = self._torso_center_and_scale(pose)
        return (pose - center) / scale

    def _velocity(self, pose: np.ndarray) -> float:
        wrists = pose[[LEFT_WRIST, RIGHT_WRIST]][:, :2]
        if self.prev_wrists is None:
            self.prev_wrists = wrists
            return 0.0
        v = float(np.linalg.norm(wrists - self.prev_wrists, ord=2))
        self.prev_wrists = wrists
        return v

    def _in_neutral(self, pose: np.ndarray) -> bool:
        shoulders = pose[[LEFT_SHOULDER, RIGHT_SHOULDER]]
        y_mid = (shoulders[0, 1] + shoulders[1, 1]) / 2.0
        wrists = pose[[LEFT_WRIST, RIGHT_WRIST]]
        return bool(np.all(wrists[:, 1] > y_mid + self.cfg.neutral_offset))

    def _clear_buffer(self) -> None:
        self.buffer.clear()
        self.prev_wrists = None
        self.gs.neutral_counter = 0
        self.state = "IDLE"
        self.gs.status = "Memory Cleared (Undo)"
        if self.gs.last_upsert_id:
            # Undo last personalized memory if available
            self.db.delete_point(self.gs.last_upsert_id)
            self.gs.last_upsert_id = None

    def _submit_search(self, vector: List[float], dialect: str, callback: Optional[Callable[[List], None]] = None) -> None:
        def _task():
            self.gs.status = "Searching Memory"
            flt = Filter(must=[FieldCondition(key="dialect", match=MatchValue(value=dialect))]) if dialect else None
            res = self.db.client.search(
                collection_name=self.db.collection_name,
                query_vector=("hand_motion", vector),
                limit=self.cfg.search_top_k,
                query_filter=flt,
                with_payload=True,
            )
            if callback:
                callback(res)
            with self._lock:
                self.gs.last_results = res
                self.gs.status = "Waiting"
        self._executor.submit(_task)

    def process_frame(
        self,
        pose_landmarks: np.ndarray,
        dialect: str,
        callback: Optional[Callable[[List], None]] = None,
    ) -> None:
        """Consume pose landmarks for a single frame.

        - Maintains sliding buffer
        - Runs state machine for motion start/end
        - Detects swipe interrupts
        - Dispatches async search on motion boundary
        """
        if pose_landmarks is None:
            return

        norm_pose = self._normalize(pose_landmarks)
        if self.swipe_handler.detect_undo_swipe(norm_pose):
            self._clear_buffer()
            return

        self.buffer.append(norm_pose)
        velocity = self._velocity(norm_pose)

        if self.state == "IDLE" and velocity > self.cfg.velocity_start:
            self.state = "RECORDING"
            self.gs.status = "Recording Gesture"
            self.gs.neutral_counter = 0

        if self.state == "RECORDING":
            if self._in_neutral(norm_pose):
                self.gs.neutral_counter += 1
            else:
                self.gs.neutral_counter = 0

            if self.gs.neutral_counter >= self.cfg.neutral_frames and len(self.buffer) >= self.cfg.min_frames:
                self.state = "PROCESSING"

        if self.state == "PROCESSING":
            seq = list(self.buffer)
            vector = self.encoder.encode_motion(seq)
            self.gs.last_vector = vector
            self.buffer.clear()
            self.state = "IDLE"
            self.gs.status = "Searching Memory"
            if vector:
                self._submit_search(vector, dialect, callback)
            else:
                self.gs.status = "Waiting"

    def register_upsert(self, point_id: str) -> None:
        self.gs.last_upsert_id = point_id

    def get_status(self) -> str:
        return self.gs.status

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
