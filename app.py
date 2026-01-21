"""Streamlit app for EchoMind real-time sign-to-speech demo with controller wiring."""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import diskcache
import mediapipe as mp
from mediapipe import solutions as mp_solutions
from mediapipe.solutions import holistic as mp_holistic
from mediapipe.solutions import drawing_utils as mp_drawing
from mediapipe.solutions import drawing_styles as mp_styles
import numpy as np
import streamlit as st
from google.cloud import texttospeech

from core.controller import GestureController
from core.controller import SignEncoder, SwipeHandler
from core.gesture_processor import LEFT_WRIST, RIGHT_WRIST
from database.qdrant_manager import EchoMindDB

CACHE_DIR = Path("cache/audio")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class GoogleTTSCache:
    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = texttospeech.TextToSpeechClient()
        self.cache = diskcache.Cache(str(cache_dir / "diskcache"))

    def _key(self, text: str, voice: str, language: str) -> str:
        h = hashlib.sha256(f"{text}|{voice}|{language}".encode()).hexdigest()
        return h

    def synthesize(self, text: str, voice: str = "en-US-Neural2-A", language: str = "en-US") -> bytes:
        cache_key = self._key(text, voice, language)
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        input_text = texttospeech.SynthesisInput(text=text)
        voice_sel = texttospeech.VoiceSelectionParams(language_code=language, name=voice)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        resp = self.client.synthesize_speech(input=input_text, voice=voice_sel, audio_config=audio_config)
        audio_bytes = resp.audio_content
        self.cache.set(cache_key, audio_bytes)
        return audio_bytes


class SimpleSignEncoder(SignEncoder):
    """Placeholder motion encoder that outputs 128-dim vectors."""

    def encode_motion(self, landmarks_seq: List[np.ndarray]) -> Optional[List[float]]:
        if not landmarks_seq:
            return None
        arr = np.stack(landmarks_seq, axis=0)
        mean_pose = arr.mean(axis=0).flatten()
        if mean_pose.size >= 128:
            vec = mean_pose[:128]
        else:
            vec = np.pad(mean_pose, (0, 128 - mean_pose.size), constant_values=0)
        return vec.astype(np.float32).tolist()


class SimpleSwipeHandler(SwipeHandler):
    """Dominant-hand swipe detector with hysteresis and cooldown."""

    def __init__(
        self,
        dx_threshold: float = 0.15,
        velocity_threshold: float = 0.05,
        cooldown_frames: int = 8,
        dominant_hand: str = "right",
    ) -> None:
        self.dx_threshold = dx_threshold
        self.velocity_threshold = velocity_threshold
        self.cooldown_frames = cooldown_frames
        self.cooldown = 0
        self.dominant_hand = dominant_hand
        self.prev_pose: Optional[np.ndarray] = None

    def _select_wrist(self, pose_landmarks: np.ndarray) -> np.ndarray:
        if self.dominant_hand == "left":
            return pose_landmarks[LEFT_WRIST, :2]
        if self.dominant_hand == "auto" and self.prev_pose is not None:
            dx_left = pose_landmarks[LEFT_WRIST, 0] - self.prev_pose[LEFT_WRIST, 0]
            dx_right = pose_landmarks[RIGHT_WRIST, 0] - self.prev_pose[RIGHT_WRIST, 0]
            return pose_landmarks[LEFT_WRIST, :2] if abs(dx_left) > abs(dx_right) else pose_landmarks[RIGHT_WRIST, :2]
        return pose_landmarks[RIGHT_WRIST, :2]

    def detect_undo_swipe(self, pose_landmarks: np.ndarray) -> bool:
        if pose_landmarks.shape[0] <= RIGHT_WRIST:
            return False
        if self.cooldown > 0:
            self.cooldown -= 1
            self.prev_pose = pose_landmarks
            return False

        wrist = self._select_wrist(pose_landmarks)
        if wrist is None:
            self.prev_pose = pose_landmarks
            return False

        if self.prev_pose is None:
            self.prev_pose = pose_landmarks
            return False

        prev_wrist = self._select_wrist(self.prev_pose)
        delta = wrist - prev_wrist
        dx, dy = delta[0], delta[1]
        speed = float(np.linalg.norm(delta, ord=2))

        self.prev_pose = pose_landmarks

        if dx > self.dx_threshold and speed > self.velocity_threshold and abs(dy) < dx * 0.5:
            self.cooldown = self.cooldown_frames
            return True
        return False


def draw_landmarks(frame: np.ndarray, holistic_res) -> np.ndarray:
    annotated = frame.copy()
    if holistic_res.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            holistic_res.pose_landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )
    if holistic_res.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            holistic_res.left_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
        )
    if holistic_res.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            holistic_res.right_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
        )
    return annotated


def main() -> None:
    st.set_page_config(page_title="EchoMind", layout="wide")
    st.title("EchoMind: Sign-to-Speech Demo")

    dialect = st.sidebar.selectbox("Dialect", options=["ASL", "ISL"], index=0)
    run_capture = st.sidebar.toggle("Run Capture", value=True)
    voice_name = st.sidebar.text_input("Voice", value="en-US-Neural2-A")
    st.sidebar.markdown("**Swipe Settings**")
    dominant_hand = st.sidebar.selectbox("Dominant Hand", options=["right", "left", "auto"], index=0)
    dx_threshold = st.sidebar.slider("Swipe dx threshold", 0.05, 0.4, 0.15, 0.01)
    velocity_threshold = st.sidebar.slider("Swipe velocity threshold", 0.01, 0.3, 0.05, 0.01)
    cooldown_frames = st.sidebar.slider("Swipe cooldown (frames)", 0, 20, 8, 1)
    status_placeholder = st.sidebar.empty()

    @st.cache_resource
    def get_services(d_hand: str, dx_thr: float, vel_thr: float, cooldown: int):
        db = EchoMindDB()
        encoder = SimpleSignEncoder()
        swipe = SimpleSwipeHandler(
            dominant_hand=d_hand,
            dx_threshold=dx_thr,
            velocity_threshold=vel_thr,
            cooldown_frames=cooldown,
        )
        controller = GestureController(encoder=encoder, swipe_handler=swipe, db=db)
        holistic = mp_holistic.Holistic()
        tts = GoogleTTSCache()
        return db, controller, holistic, tts

    db, controller, holistic, tts = get_services(dominant_hand, dx_threshold, velocity_threshold, cooldown_frames)

    video_placeholder = st.empty()
    results_placeholder = st.empty()

    st.session_state.setdefault("results_hits", [])

    def on_results(hits) -> None:
        st.session_state["results_hits"] = hits

    if not run_capture:
        st.info("Toggle 'Run Capture' to start the webcam feed.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access webcam.")
        return

    try:
        while run_capture:
            ok, frame = cap.read()
            if not ok:
                st.warning("Camera read failed.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            annotated = draw_landmarks(rgb, res)

            pose_arr = None
            if res.pose_landmarks:
                pose_arr = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark], dtype=np.float32)

            controller.process_frame(pose_arr, dialect, callback=on_results)

            status = controller.get_status()
            status_placeholder.write(f"Status: {status}")
            if "Memory Cleared" in status:
                st.session_state["results_hits"] = []

            hits = st.session_state.get("results_hits", [])
            if hits:
                with results_placeholder.container():
                    st.subheader("Top Candidates")
                    cols = st.columns(len(hits))
                    for col, hit in zip(cols, hits):
                        gloss = hit.payload.get("gloss", "")
                        score = round(hit.score, 3) if hit.score is not None else 0
                        label = f"{gloss} ({score})"
                        if col.button(label):
                            payload = hit.payload or {}
                            payload["dialect"] = dialect
                            vector = controller.gs.last_vector or []
                            if vector:
                                pid = db.upsert_correction({"hand_motion": vector}, payload, point_id=str(hit.id))
                                controller.register_upsert(pid)
                            audio = tts.synthesize(gloss, voice=voice_name)
                            st.audio(io.BytesIO(audio), format="audio/mp3")

            video_placeholder.image(annotated, channels="RGB")
            time.sleep(0.01)

    finally:
        cap.release()
        controller.shutdown()
        holistic.close()


if __name__ == "__main__":
    main()
