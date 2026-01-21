# EchoMind Qdrant Cloud & Piper TTS Integration Guide

## Overview
EchoMind has been refactored to use **Qdrant Cloud** for vector storage and **Piper TTS** for offline neural speech synthesis.

---

## 1. Qdrant Cloud Setup

### Environment Variables
Add these to your `.env` file (already configured):

```env
# Qdrant Cloud Connection (priority)
QDRANT_URL="https://22ab3797-0f2e-4cb4-84b9-9e82474a4c69.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Q-oODCsy17uduR_4vvkhC-YyAwP5MU8VPOxbT8WdufM"

# Qdrant Local Server (fallback if cloud unavailable)
QDRANT_HOST="localhost"
QDRANT_PORT=6333
```

### Connection Priority
1. **Cloud URL** (`QDRANT_URL` + `QDRANT_API_KEY`) - Primary
2. **Local Server** (`QDRANT_HOST` + `QDRANT_PORT`) - Fallback
3. **In-Memory** - Last resort if both fail

### Key Features
- ✅ Auto-appends `:6333` port if not present in URL
- ✅ Connection tested on init with `get_collections()`
- ✅ Cached in Streamlit via `@st.cache_resource` to prevent reconnections
- ✅ Graceful fallback to in-memory database if cloud/local unavailable

---

## 2. Search & Retrieval

### New `search_gesture()` Method
Located in `database/qdrant_manager.py`:

```python
results = db.search_gesture(
    query_vector=[0.1] * 256,  # 256-dim hand motion vector
    dialect='ASL',              # Filter by dialect
    vector_name='hand_motion',  # Which named vector to search
    limit=5                     # Top-K results
)
```

**Features:**
- **Cosine Similarity** - Default distance metric
- **Dialect Filtering** - Only returns gestures matching ASL/ISL/etc.
- **Named Vector Support** - Search `hand_motion` or `face_expression`
- **Returns** - List of `ScoredPoint` objects with scores and payloads

### Usage in Controller
The `GestureController` now uses `search_gesture()` instead of direct client calls:

```python
# core/controller.py - _submit_search() method
res = self.db.search_gesture(
    query_vector=vector,
    dialect=dialect,
    vector_name="hand_motion",
    limit=self.cfg.search_top_k,
)
```

---

## 3. Piper TTS Integration

### Environment Variables
Add to `.env` (already configured):

```env
# Piper TTS model paths
PIPER_MODEL=./models/en_US-amy-medium.onnx
PIPER_CONFIG=./models/en_US-amy-medium.onnx.json
PIPER_USE_CUDA=false
```

### Download Voices
Get voices from: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md

Example:
```bash
# Create models directory
mkdir models

# Download a voice (example: en_US-amy-medium)
# Place .onnx and .onnx.json files in models/
```

### Cached Model Loading
Located in `app.py`:

```python
@st.cache_resource(show_spinner=False)
def _load_piper_voice(model_path: str, config_path: str, use_cuda: bool = False):
    from piper.voice import PiperVoice
    return PiperVoice.load(model_path, config_path, use_cuda=use_cuda)
```

**Features:**
- ✅ Model loaded **once** on first synthesis
- ✅ Cached across Streamlit reruns
- ✅ Automatic fallback to **pyttsx3** if Piper unavailable
- ✅ Outputs **WAV bytes** compatible with `st.audio` and DiskCache

### TTS Class Structure
```python
class PiperTTSCache:
    def synthesize(self, text: str, voice: str = "", language: str = "") -> bytes:
        # 1. Check DiskCache
        # 2. Try Piper synthesis → WAV bytes
        # 3. Fallback to pyttsx3 → WAV bytes
        # 4. Cache result and return
```

---

## 4. Personalization Loop (Reinforcement)

### `upsert_correction()` Method
Located in `database/qdrant_manager.py`:

```python
point_id = db.upsert_correction(
    vectors={'hand_motion': vector_128_or_256},
    payload={'gloss': 'hello', 'dialect': 'ASL', 'success_count': 0},
    point_id='optional-uuid',  # Auto-generated if None
    increment_success=True      # Bumps success_count by +1
)
```

**Features:**
- ✅ Increments `success_count` for RL when `increment_success=True`
- ✅ Upserts to Qdrant Cloud with `wait=True` for immediate consistency
- ✅ Returns `point_id` for undo/tracking

### Usage in App
When user confirms a gesture via button click:

```python
if col.button(label):
    # Update Qdrant with confirmed gloss
    pid = db.upsert_correction(
        {'hand_motion': vector},
        {'gloss': gloss, 'dialect': dialect, 'success_count': hit.payload.get('success_count', 0)},
        point_id=str(hit.id),
        increment_success=True  # +1 to success_count
    )
    controller.register_upsert(pid)
```

---

## 5. Testing

### Verify Qdrant Cloud Connection
```python
from database.qdrant_manager import EchoMindDB
db = EchoMindDB()
# Should print: "Connected to Qdrant Cloud: https://..."
```

### Test Search
```python
results = db.search_gesture([0.1]*256, 'ASL', limit=3)
print(f"Found {len(results)} results")
```

### Test Upsert with Increment
```python
test_id = db.upsert_correction(
    {'hand_motion': [0.1]*256},
    {'gloss': 'test', 'dialect': 'ASL', 'success_count': 0},
    increment_success=True
)
# success_count will be 1 in Qdrant
```

---

## 6. Running the App

### Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Launch Streamlit
```powershell
python -m streamlit run app.py
```

### Expected Console Output
```
Connected to Qdrant Cloud: https://22ab3797-0f2e-4cb4-84b9-9e82474a4c69...
Local URL: http://localhost:8501
```

---

## 7. Troubleshooting

### Qdrant Connection Fails
- Verify `QDRANT_URL` includes `:6333` port
- Check `QDRANT_API_KEY` is correct
- Falls back to in-memory automatically (prints warning)

### Piper Voice Not Found
- Verify `PIPER_MODEL` and `PIPER_CONFIG` paths exist
- Falls back to pyttsx3 automatically (no error)

### Protobuf Conflicts
- Ensure `protobuf>=4.25.3,<5` in requirements.txt
- Run: `pip install --upgrade "protobuf>=4.25.3,<5"`

---

## 8. Files Modified

### Core Changes
- `database/qdrant_manager.py` - Cloud connection + `search_gesture()` + `upsert_correction()`
- `core/controller.py` - Updated to use `search_gesture()`
- `app.py` - Piper TTS with `@st.cache_resource` + WAV output
- `.env` - Added cloud credentials + Piper paths
- `requirements.txt` - Added `piper-tts`, `python-dotenv`, pinned `protobuf`

### Dependencies Added
- `piper-tts` - Offline neural TTS
- `python-dotenv` - Environment variable loading
- `protobuf>=4.25.3,<5` - Mediapipe compatibility

---

## Summary

✅ **Qdrant Cloud** - Auto-connects with cloud URL priority  
✅ **Search with Filters** - Cosine similarity + dialect filtering  
✅ **Piper TTS** - Cached offline neural speech (WAV output)  
✅ **Reinforcement** - `upsert_correction()` increments `success_count`  
✅ **Streamlit Caching** - Both DB and TTS model cached via `@st.cache_resource`
