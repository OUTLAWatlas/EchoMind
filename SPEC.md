# EchoMind Specification

## 1. Goal & Scope
EchoMind is a personalized, real-time sign-to-speech engine for the Convolve 4.0 "Accessibility & Inclusion" challenge. It ingests continuous sign language video, produces spatiotemporal embeddings, retrieves glosses via Qdrant named vectors, and outputs natural speech with adaptive personalization and reinforcement.

## 2. System Overview
- **Capture**: Webcam → MediaPipe Holistic (hands, pose, face) at ~30 FPS.
- **Segmentation (Adaptive Windowing)**: Motion-start and neutral-zone detection based on wrist velocity relative to shoulders.
- **Encoding**: Sliding/variable window → temporal encoder (LSTM/Conformer) → named vectors.
- **Retrieval & Memory (Qdrant)**: HNSW + scalar quantization on named vectors (`hand_motion`, `face_expression`; optionally `body_pose`) with payload filters for dialect/context; reinforcement via upsert on corrections.
- **Ranking & Traceability**: Top-k with similarity scores and point IDs surfaced to UI for explainability.
- **Speech**: Hybrid TTS (Google Cloud TTS primary) with local cache for common glosses; pyttsx3 offline fallback.
- **UX Loop**: Streamlit overlay with dialect toggle, top-3 candidates, correction buttons (upsert), swipe-to-delete gesture (clear buffer/delete), and payload-based filtering.

## 3. Data Flow (Happy Path)
1) Frames captured.
2) Landmarks extracted (hands/pose/face) → normalized coordinates.
3) Adaptive window start when hand velocity exceeds threshold; end after neutral-zone dwell.
4) Windowed landmarks → temporal encoder → named vectors per modality.
5) Qdrant query with payload filter (`dialect`, optional `setting`) → top-k.
6) UI shows top-3 candidates with similarity + point IDs.
7) User accepts or selects alternate; alternate triggers upsert (reinforcement) and `success_count` increment.
8) Gloss text routed to TTS cache; cache hit plays immediately; miss triggers Google TTS → cache store → play. Fallback to pyttsx3 on failure.
9) Swipe gesture clears pending buffer or deletes last point (if mapped).

## 4. Adaptive Windowing Logic
- **Velocity start**: Let $v_t$ be summed wrist velocity (norm units/frame). Start recording when $v_t > 0.05$ for either hand.
- **Neutral zone end**: Pose landmarks 11/12 (shoulders) define mid-shoulder plane: $y_{mid} = (y_{11} + y_{12}) / 2$. If both wrists $y_{wrist} > y_{mid} + 0.15$ for $\ge 5$ consecutive frames, close window.
- **Debounce**: Minimum window length 8 frames; maximum 120 frames before forced close.
- **Buffering**: Keep last 2 closed windows for correction/backspace operations.

## 5. Vector Strategy (Qdrant)
- **Collections**: Single collection `echomind_signs` using named vectors.
- **Named vectors**:
  - `hand_motion`: primary spatiotemporal embedding of hand landmarks.
  - `face_expression`: facial/eyebrow cues embedding.
  - Optional `body_pose`: coarse pose context for robustness.
- **Indexing**: HNSW per named vector; metric `cosine`; `ef_construct` tuned for build; runtime `ef` tuned for latency vs recall.
- **Compression**: Scalar Quantization enabled for sub-millisecond retrieval and reduced memory.
- **Filtering**: Payload filters for `dialect`, `setting`, optional `priority` and `success_count` boosts.

## 6. Payload & Metadata Schema
Canonical payload per point:
```json
{
  "gloss": "HELP",
  "category": "emergency",
  "dialect": "ASL",
  "setting": "general",  
  "priority": 1.0,
  "success_count": 42,
  "created_at": "2026-01-18T00:00:00Z",
  "updated_at": "2026-01-18T00:00:00Z",
  "source": "user|seed|mock",
  "notes": "optional free text"
}
```
- **Boosting**: Rank by similarity, then adjust with `success_count` and `priority` (e.g., score *= 1 + log1p(success_count)).
- **Dialect demo**: ASL vs ISL variants stored as separate points with `dialect` tags; UI toggle switches filter.

## 7. TTS Strategy
- **Primary**: Google Cloud TTS via ADC; voices configured for naturalness; rate/pitch adjustable for clarity.
- **Cache**: Pre-generate and store WAV/MP3 for high-frequency glosses (hello, my name, yes, no, please, thank you, water, food, drink, bathroom, help, emergency, danger, hurt/pain, sick, doctor, ambulance, stop, stay, wait, go, namaste).
- **Fallback**: pyttsx3 when cloud unavailable; log and optionally enqueue cloud retry to refresh cache.

## 8. UX & Reinforcement Loop
- **Streamlit overlay**: Camera preview, dialect toggle, top-3 candidate buttons with similarity and point ID.
- **Correction**: Selecting a non-top1 candidate triggers `collection.upsert()` with the window’s vectors and corrected payload; increments `success_count` for that gloss/dialect.
- **Deletion/Undo**: Left-to-right dominant-hand swipe clears current buffer; may map to delete last upsert (if enabled) using `collection.delete()` by point ID or user buffer.
- **Traceability**: Display point ID and similarity score for transparency.

## 9. Security / DevSecOps
- **Secrets**: ADC via `GOOGLE_APPLICATION_CREDENTIALS`; no keys committed. Use `.env` (ignored) for Qdrant host/port.
- **Network**: TLS for Qdrant if remote; restrict ingress by IP; prefer API key/token if available.
- **Least privilege**: GCP service account scoped to Text-to-Speech only; no broad roles.
- **Supply chain**: Pin dependencies (requirements.txt); enable hashing (pip-tools/uv) if time.
- **Observability**: Structured logs for window events, retrieval latency, TTS cache hits/misses, upsert outcomes, and errors. Basic health endpoint for UI.
- **Privacy**: Do not persist raw video; store only embeddings + minimal payload; allow user to purge personalized entries.

## 10. Proposed File Tree (scaffold)
- `SPEC.md` (this spec)
- `prd.txt` (product brief)
- `src/`
  - `capture/` (camera + MediaPipe wrapper)
  - `windowing/` (adaptive window detector)
  - `encoding/` (temporal encoder to named vectors)
  - `qdrant_client/` (client, schema bootstrap, search, upsert, delete)
  - `tts/` (google TTS client, cache manager, pyttsx3 fallback)
  - `ui/streamlit_app.py` (overlay, buttons, swipe handling)
  - `config.py` (env parsing, constants)
- `data/mock_samples/` (seed ASL/ISL exemplars, metadata)
- `cache/audio/` (pre-rendered gloss audio)
- `.env.example` (GOOGLE_APPLICATION_CREDENTIALS, QDRANT_HOST/PORT)
- `credentials.example.json` (service account template)
- `requirements.txt`

## 11. Implementation Plan (10 steps)
1) **Env + deps**: Set up Python env; add requirements (mediapipe, opencv-python, numpy, torch or tf (for encoder), qdrant-client, google-cloud-texttospeech, streamlit, pyttsx3, cachetools).
2) **Config plumbing**: Implement `config.py` to read env (ADC path, Qdrant host/port, cache dirs) and validate.
3) **Qdrant schema bootstrap**: Define named vectors, HNSW params, enable scalar quantization; create collection if missing.
4) **Windowing module**: Implement velocity start ($v_t > 0.05$) and neutral-zone end (wrists below shoulder midline + 0.15 for ≥5 frames); add debouncing and buffering.
5) **Encoder**: Build temporal encoder for windows (start with lightweight LSTM/Conformer); output per-modality vectors (`hand_motion`, `face_expression`; optional `body_pose`).
6) **Mock data seeding**: Ingest ASL/ISL paired exemplars (greetings/emergency) with payload schema; set `priority` and initial `success_count`.
7) **Retrieval service**: Wrap search with payload filters (dialect/setting), scoring with similarity + `success_count` boost; return point IDs and scores.
8) **Reinforcement loop**: Implement upsert on correction; increment `success_count`; support delete/undo by point ID or buffer.
9) **TTS service**: Add cache-first lookup; Google TTS on miss; store audio in `cache/audio`; pyttsx3 fallback with logging.
10) **Streamlit UI**: Camera preview, dialect toggle, top-3 buttons, similarity + point ID display, swipe-to-delete handler; wire to retrieval and reinforcement.

## 12. Risks & Mitigations
- **Latency**: Use scalar quantization, tuned `ef_search`, and cache common TTS outputs; pre-load models on start.
- **Segmentation errors**: Log window stats; adjust velocity/offset thresholds; add optional hysteresis.
- **Data sparsity**: Seed with mocked ASL/ISL pairs; prioritize reinforcement loop to personalize quickly.
- **Connectivity**: Offline fallback for TTS; local Qdrant option.

## 13. Success Criteria
- End-to-end latency target: <300 ms retrieval; audio playback <150 ms for cached glosses.
- Accurate dialect toggle: Same motion yields different gloss when switching ASL ↔ ISL.
- Reinforcement: Correction immediately upserts and is preferred on subsequent queries.
- Transparency: UI shows point ID and similarity for each candidate.
