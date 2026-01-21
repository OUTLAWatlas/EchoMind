"""Streamlit app for EchoMind real-time sign-to-speech demo with Qdrant Cloud + Piper TTS."""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
import os
import wave
import tempfile
from typing import Dict, List, Optional

import cv2
import diskcache
import mediapipe as mp
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_styles
import numpy as np
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from core.controller import GestureController
from core.controller import SignEncoder, SwipeHandler
from core.gesture_processor import LEFT_WRIST, RIGHT_WRIST
from database.qdrant_manager import EchoMindDB

CACHE_DIR = Path("cache/audio")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "echomind_signs"


@st.cache_resource(show_spinner=False)
def _load_piper_voice(model_path: str, config_path: str, use_cuda: bool = False):
    """Load Piper voice model once and cache across reruns."""
    from piper.voice import PiperVoice
    return PiperVoice.load(model_path, config_path, use_cuda=use_cuda)


@st.cache_resource(show_spinner=False)
def _connect_qdrant_cloud() -> QdrantClient:
    """Establish Qdrant Cloud connection with environment credentials.
    
    Cached to ensure connection persists across Streamlit reruns.
    Includes graceful error handling for cloud latency or connection drops.
    """
    try:
        url = os.getenv("QDRANT_URL", "")
        api_key = os.getenv("QDRANT_API_KEY", "")
        
        if not url or not api_key:
            st.warning("QDRANT_URL and QDRANT_API_KEY not set in .env")
            return None
        
        # Auto-append port if not present
        if ":6333" not in url:
            url = f"{url}:6333"
        
        client = QdrantClient(url=url, api_key=api_key, timeout=30.0)
        # Test connection
        client.get_collection(COLLECTION_NAME)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Qdrant Cloud: {e}")
        return None



class PiperTTSCache:
    """Offline neural TTS using Piper with disk caching and auto-playback.

    Features:
    - Loads Piper ONNX/JSON models once via @st.cache_resource
    - Disk-caches synthesized audio using DiskCache
    - Returns WAV bytes compatible with st.audio()
    - Falls back to pyttsx3 if Piper unavailable
    - Integrates with st.session_state for playback tracking
    """

    def __init__(self, cache_dir: Path = CACHE_DIR,
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 use_cuda: bool = False) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(cache_dir / "diskcache"))

        # Load from environment or use provided paths
        self.model_path = model_path or os.getenv("PIPER_MODEL", "")
        self.config_path = config_path or os.getenv("PIPER_CONFIG", "")
        self.use_cuda = use_cuda or (os.getenv("PIPER_USE_CUDA", "false").lower() == "true")

    def _key(self, text: str) -> str:
        """Generate cache key from model path and text."""
        ident = f"{self.model_path}|{self.config_path}|{text}"
        return hashlib.sha256(ident.encode()).hexdigest()

    def _synthesize_with_piper(self, text: str) -> Optional[bytes]:
        """Synthesize audio using Piper ONNX model."""
        if not self.model_path or not self.config_path:
            print(f"[PIPER] Model paths not set: model={self.model_path}, config={self.config_path}")
            return None
        try:
            print(f"[PIPER] Loading voice from {self.model_path}")
            voice = _load_piper_voice(self.model_path, self.config_path, self.use_cuda)
            
            # Get sample rate from voice config
            sample_rate = 22050
            try:
                if hasattr(voice, 'config') and hasattr(voice.config, 'audio'):
                    sample_rate = voice.config.audio.sample_rate
                elif hasattr(voice, 'config') and hasattr(voice.config, 'sample_rate'):
                    sample_rate = voice.config.sample_rate
            except:
                pass
            
            print(f"[PIPER] Synthesizing '{text}' at {sample_rate}Hz")
            
            # Use synthesize_wav which returns WAV bytes directly
            audio_bytes = voice.synthesize_wav(text)
            
            print(f"[PIPER] Generated {len(audio_bytes)} bytes")
            
            # Check if we got real audio (more than just WAV header)
            if len(audio_bytes) <= 100:
                print(f"[PIPER] Warning: Audio too small ({len(audio_bytes)} bytes), likely empty")
                return None
            
            return audio_bytes
        except Exception as e:
            print(f"[PIPER ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _synthesize_with_pyttsx3(self, text: str) -> Optional[bytes]:
        """Fallback synthesize using pyttsx3."""
        try:
            print(f"[PYTTSX3] Synthesizing '{text}'")
            import pyttsx3
            engine = pyttsx3.init()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                temp_path = tf.name
            try:
                engine.save_to_file(text, temp_path)
                engine.runAndWait()
                with open(temp_path, "rb") as f:
                    audio_bytes = f.read()
                print(f"[PYTTSX3] Generated {len(audio_bytes)} bytes")
                return audio_bytes
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"[PYTTSX3 ERROR] {e}")
            return None

    def synthesize(self, text: str, voice: str = "", language: str = "") -> bytes:
        """Synthesize text to audio bytes with disk caching.
        
        Args:
            text: Text to synthesize
            voice: Voice identifier (currently unused, for future expansion)
            language: Language code (currently unused, for future expansion)
        
        Returns:
            WAV audio bytes suitable for st.audio()
        """
        # Skip cache for now to test Piper
        print(f"[TTS] Synthesizing '{text}'...")
        # Try Piper first, fall back to pyttsx3
        audio_bytes = self._synthesize_with_piper(text)
        if audio_bytes is None:
            print(f"[TTS] Piper failed, trying pyttsx3...")
            audio_bytes = self._synthesize_with_pyttsx3(text)
        if audio_bytes is None:
            # Last resort: empty WAV (silence)
            print(f"[TTS] Both engines failed, returning silence")
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(b"\x00" * 2205)  # ~0.05s silence
            audio_bytes = buf.getvalue()

        print(f"[TTS] Final audio: {len(audio_bytes)} bytes")
        return audio_bytes


class SimpleSignEncoder(SignEncoder):
    """Motion encoder that outputs 256-dim vectors matching Qdrant collection schema."""

    def encode_motion(self, landmarks_seq: List[np.ndarray]) -> Optional[List[float]]:
        if not landmarks_seq:
            return None
        arr = np.stack(landmarks_seq, axis=0)
        mean_pose = arr.mean(axis=0).flatten()
        
        # Pad or truncate to 256 dimensions to match Qdrant schema
        if mean_pose.size >= 256:
            vec = mean_pose[:256]
        else:
            vec = np.pad(mean_pose, (0, 256 - mean_pose.size), constant_values=0)
        
        # Normalize vector for cosine similarity search
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
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
    """Draw MediaPipe landmarks (pose, hands) on video frame."""
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


def display_lexicon_explorer(client: Optional[QdrantClient]) -> None:
    """Display Lexicon Explorer: all available signs in Qdrant Cloud.
    
    Uses client.scroll() to efficiently retrieve all unique signs (glosses)
    from the vector database, allowing users to explore what's available.
    """
    if client is None:
        st.warning("Lexicon Explorer: Qdrant connection not available")
        return
    
    try:
        with st.expander("Lexicon Explorer", expanded=False):
            st.markdown("**All Available Signs in Database**")
            
            # Scroll through collection to collect unique signs
            signs_dict: Dict[str, int] = {}  # gloss -> count
            
            limit = 100  # Batch size for scrolling
            offset = None
            
            while True:
                try:
                    points, next_offset = client.scroll(
                        collection_name=COLLECTION_NAME,
                        limit=limit,
                        offset=offset,
                        with_payload=True,
                    )
                    
                    if not points:
                        break
                    
                    for point in points:
                        gloss = point.payload.get("gloss", "Unknown")
                        signs_dict[gloss] = signs_dict.get(gloss, 0) + 1
                    
                    offset = next_offset
                    if next_offset is None:
                        break
                except Exception as scroll_err:
                    st.warning(f"Error scrolling collection: {scroll_err}")
                    break
            
            if signs_dict:
                # Display as a searchable list
                sorted_signs = sorted(signs_dict.items(), key=lambda x: -x[1])
                
                cols = st.columns(3)
                for idx, (gloss, count) in enumerate(sorted_signs):
                    col = cols[idx % 3]
                    col.metric(gloss, f"{count} variants")
                
                st.info(f"Total unique signs: {len(signs_dict)}")
            else:
                st.info("No signs found in database yet.")
    except Exception as e:
        st.error(f"Lexicon Explorer error: {e}")



def main() -> None:
    """Main Streamlit application with Qdrant Cloud + Piper TTS integration.
    
    Architecture:
    - QdrantClient cached via @st.cache_resource for persistent cloud connection
    - st.session_state manages translation history, audio playback state, and buffer
    - Dialect-based search filtering via Qdrant Filter API
    - Automatic TTS synthesis and playback when sign recognized
    - Lexicon Explorer for browsing available signs
    """
    st.set_page_config(page_title="EchoMind", layout="wide")
    st.title("EchoMind")

    # ==================== Sidebar Configuration ====================
    with st.sidebar:
        st.markdown("## Configuration")
        
        # Dialect selection for search filtering
        dialect = st.selectbox("Dialect", options=["ASL", "ISL"], index=0)
        
        # Capture control
        run_capture = st.toggle("Run Capture", value=True)
        
        # TTS voice settings
        st.markdown("### TTS Voice")
        voice_name = st.text_input("Voice Name", value="en-US-Neural2-A")
        
        # Gesture detection settings
        st.markdown("### Swipe/Undo Settings")
        dominant_hand = st.selectbox("Dominant Hand", options=["right", "left", "auto"], index=0)
        dx_threshold = st.slider("Swipe dx threshold", 0.05, 0.4, 0.15, 0.01)
        velocity_threshold = st.slider("Swipe velocity threshold", 0.01, 0.3, 0.05, 0.01)
        cooldown_frames = st.slider("Swipe cooldown (frames)", 0, 20, 8, 1)
        
        # Status display
        st.markdown("### Status")
        status_placeholder = st.empty()

    # ==================== Service Initialization ====================
    @st.cache_resource
    def get_services(d_hand: str, dx_thr: float, vel_thr: float, cooldown: int):
        """Initialize cached services: DB, Controller, TTS, Qdrant."""
        db = EchoMindDB()
        encoder = SimpleSignEncoder()
        swipe = SimpleSwipeHandler(
            dominant_hand=d_hand,
            dx_threshold=dx_thr,
            velocity_threshold=vel_thr,
            cooldown_frames=cooldown,
        )
        controller = GestureController(encoder=encoder, swipe_handler=swipe, db=db)
        tts = PiperTTSCache()
        qdrant_client = _connect_qdrant_cloud()
        return db, controller, tts, qdrant_client

    db, controller, tts, qdrant_client = get_services(
        dominant_hand, dx_threshold, velocity_threshold, cooldown_frames
    )
    
    # Create MediaPipe Holistic fresh (not cached to avoid graph lifecycle issues)
    holistic = mp_holistic.Holistic()

    # ==================== Session State Management ====================
    # Initialize session state for translation history and playback control
    st.session_state.setdefault("results_hits", [])
    st.session_state.setdefault("current_translation", "")
    st.session_state.setdefault("last_audio_gloss", "")
    st.session_state.setdefault("buffer_sequence", [])

    # ==================== UI Layout ====================
    video_placeholder = st.empty()
    results_placeholder = st.empty()
    audio_placeholder = st.empty()  # Dedicated audio player
    
    # Display Lexicon Explorer
    display_lexicon_explorer(qdrant_client)

    def on_results(hits) -> None:
        """Callback when Qdrant search returns results."""
        st.session_state["results_hits"] = hits

    # ==================== Capture Loop ====================
    if not run_capture:
        st.info("Toggle 'Run Capture' in sidebar to start webcam feed.")
        if "camera" in st.session_state:
            st.session_state.camera.release()
            del st.session_state.camera
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access webcam. Check camera permissions.")
        return

    try:
        frame_count = 0
        while run_capture:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            annotated = draw_landmarks(rgb, res)

            pose_arr = None
            if res.pose_landmarks:
                pose_arr = np.array(
                    [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark],
                    dtype=np.float32
                )

            try:
                controller.process_frame(pose_arr, dialect, callback=on_results)
            except Exception as e:
                pass

            status = controller.get_status()
            status_placeholder.write(f"**Status:** {status} | Frame: {frame_count}")
            
            if "Memory Cleared" in status:
                st.session_state["results_hits"] = []

            hits = st.session_state.get("results_hits", [])
            if hits:
                with results_placeholder.container():
                    st.subheader("Top Candidates")
                    cols = st.columns(min(len(hits), 5))
                    for idx, (col, hit) in enumerate(zip(cols, hits[:5])):
                        with col:
                            gloss = hit.payload.get("gloss", "Unknown")
                            score = round(hit.score, 3) if hit.score is not None else 0
                            label = f"{gloss}\n({score})"
                            # Add frame_count to make key unique across loop iterations
                            button_key = f"btn_{hit.id}_{idx}_{frame_count}"
                            if st.button(label, key=button_key, use_container_width=True):
                                st.session_state["trigger_audio"] = gloss
                                payload = hit.payload or {}
                                payload["dialect"] = dialect
                                vector = controller.gs.last_vector or []
                                if vector:
                                    try:
                                        db.upsert_correction(
                                            {"hand_motion": vector},
                                            payload,
                                            point_id=str(hit.id),
                                            increment_success=True
                                        )
                                    except:
                                        pass

            if "trigger_audio" in st.session_state:
                gloss_to_play = st.session_state["trigger_audio"]
                try:
                    print(f"[AUDIO] Synthesizing: {gloss_to_play}")
                    print(f"[DEBUG] TTS model_path: {tts.model_path}")
                    print(f"[DEBUG] TTS config_path: {tts.config_path}")
                    audio = tts.synthesize(gloss_to_play, voice=voice_name)
                    print(f"[DEBUG] Generated audio size: {len(audio)} bytes")
                    if len(audio) > 0:
                        with audio_placeholder:
                            st.audio(io.BytesIO(audio), format="audio/wav")
                        print(f"[AUDIO] Playing: {gloss_to_play}")
                    else:
                        print(f"[ERROR] Audio is empty!")
                    del st.session_state["trigger_audio"]
                except Exception as e:
                    print(f"[AUDIO ERROR] {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    if "trigger_audio" in st.session_state:
                        del st.session_state["trigger_audio"]

            video_placeholder.image(annotated, channels="RGB")
            time.sleep(0.033)
    
    finally:
        cap.release()


if __name__ == "__main__":
    main()

