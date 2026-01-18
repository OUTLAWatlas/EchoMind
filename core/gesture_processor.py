"""Gesture processing with MediaPipe Holistic and adaptive windowing.

Extracts landmarks, applies torso-relative normalization, and segments motion
windows using velocity start and neutral-zone end conditions.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16


@dataclass
class WindowConfig:
    velocity_start: float = 0.05  # norm units/frame
    neutral_offset: float = 0.15  # added to shoulder midline y
    neutral_frames: int = 5       # frames required in neutral to close
    min_frames: int = 8
    max_frames: int = 120


@dataclass
class Window:
    frames: List[np.ndarray] = field(default_factory=list)
    landmarks: List[Dict[str, List[Tuple[float, float, float]]]] = field(default_factory=list)


class GestureProcessor:
    def __init__(self, config: Optional[WindowConfig] = None) -> None:
        self.cfg = config or WindowConfig()
        self.holistic = mp.solutions.holistic.Holistic()
        self.state: str = "IDLE"  # IDLE or ACTIVE
        self.current_window = Window()
        self.prev_wrist: Optional[np.ndarray] = None
        self.neutral_counter = 0
        self.closed_windows: Deque[Window] = collections.deque(maxlen=2)

    def _extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.holistic.process(rgb)
        if not res.pose_landmarks:
            return None
        coords = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark], dtype=np.float32)
        return coords

    def _torso_center_and_scale(self, pose: np.ndarray) -> Tuple[np.ndarray, float]:
        shoulders = pose[[LEFT_SHOULDER, RIGHT_SHOULDER]]
        center = shoulders.mean(axis=0)
        scale = np.linalg.norm(shoulders[0, :2] - shoulders[1, :2]) + 1e-6
        return center, scale

    def _normalize_landmarks(self, pose: np.ndarray) -> np.ndarray:
        center, scale = self._torso_center_and_scale(pose)
        return (pose - center) / scale

    def _wrist_velocity(self, pose: np.ndarray) -> float:
        wrists = pose[[LEFT_WRIST, RIGHT_WRIST]][:, :2]
        if self.prev_wrist is None:
            self.prev_wrist = wrists
            return 0.0
        v = np.linalg.norm(wrists - self.prev_wrist, ord=2)
        self.prev_wrist = wrists
        return float(v)

    def _in_neutral(self, pose: np.ndarray) -> bool:
        shoulders = pose[[LEFT_SHOULDER, RIGHT_SHOULDER]]
        y_mid = (shoulders[0, 1] + shoulders[1, 1]) / 2.0
        wrists = pose[[LEFT_WRIST, RIGHT_WRIST]]
        return bool(np.all(wrists[:, 1] > y_mid + self.cfg.neutral_offset))

    def _close_window_if_ready(self) -> Optional[Window]:
        if len(self.current_window.frames) < self.cfg.min_frames:
            self._reset_window()
            return None
        closed = self.current_window
        self.closed_windows.append(closed)
        self._reset_window()
        return closed

    def _reset_window(self) -> None:
        self.state = "IDLE"
        self.current_window = Window()
        self.neutral_counter = 0

    def process_frame(self, frame: np.ndarray) -> Optional[Window]:
        pose = self._extract_landmarks(frame)
        if pose is None:
            return None

        norm_pose = self._normalize_landmarks(pose)
        velocity = self._wrist_velocity(norm_pose)

        if self.state == "IDLE" and velocity > self.cfg.velocity_start:
            self.state = "ACTIVE"

        if self.state == "ACTIVE":
            self.current_window.frames.append(frame)
            self.current_window.landmarks.append({"pose": norm_pose.tolist()})

            if self._in_neutral(norm_pose):
                self.neutral_counter += 1
            else:
                self.neutral_counter = 0

            if self.neutral_counter >= self.cfg.neutral_frames or len(self.current_window.frames) >= self.cfg.max_frames:
                return self._close_window_if_ready()

        return None

    def last_closed_windows(self) -> List[Window]:
        return list(self.closed_windows)

    def close(self) -> None:
        self.holistic.close()
