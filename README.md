# EchoMind

A real-time sign language-to-speech system powered by AI, designed to bridge communication gaps for the Deaf and hard-of-hearing community. EchoMind uses computer vision, vector search, and neural text-to-speech to provide instant, personalized sign language translation.

## 🌟 Features

- **Real-time Sign Recognition**: Uses MediaPipe Holistic for accurate hand, pose, and facial landmark detection
- **Vector Search with Qdrant Cloud**: Fast, scalable gesture matching with cosine similarity
- **Dialect Support**: Toggle between ASL (American Sign Language) and ISL (Indian Sign Language)
- **Neural Text-to-Speech**: Offline speech synthesis using Piper TTS with automatic fallback to pyttsx3
- **Adaptive Personalization**: Reinforcement learning that improves accuracy based on user corrections
- **Interactive UI**: Clean Streamlit interface with live camera feed and gesture predictions
- **Swipe Gestures**: Natural undo/clear functionality with hand swipe detection

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (for real-time gesture capture)
- Internet connection (for Qdrant Cloud)

### System Dependencies

#### Windows
```powershell
# Install espeak for pyttsx3 fallback (optional)
# Download from: http://espeak.sourceforge.net/download.html
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y espeak portaudio19-dev python3-pyaudio
```

#### macOS
```bash
brew install espeak portaudio
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/OUTLAWatlas/EchoMind.git
   cd EchoMind
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root with the following configuration:
   
   ```env
   # Qdrant Cloud Connection (required)
   QDRANT_URL="https://22ab3797-0f2e-4cb4-84b9-9e82474a4c69.europe-west3-0.gcp.cloud.qdrant.io"
   QDRANT_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Q-oODCsy17uduR_4vvkhC-YyAwP5MU8VPOxbT8WdufM"
   
   # Qdrant Local Server (fallback, optional)
   QDRANT_HOST="localhost"
   QDRANT_PORT=6333
   
   # Piper TTS Model Paths (for neural speech synthesis)
   PIPER_MODEL=./models/en_US-amy-medium.onnx
   PIPER_CONFIG=./models/en_US-amy-medium.onnx.json
   PIPER_USE_CUDA=false
   ```

5. **Download Piper TTS Voice Models** (optional but recommended)
   
   Download pre-trained voice models from [Piper Voices](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md)
   
   ```bash
   mkdir -p models
   # Download .onnx and .onnx.json files to the models/ directory
   ```
   
   The models directory already contains `en_US-amy-medium.onnx` and its config file.

## 🎯 Getting Started

### Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Or use the Python module syntax:

```bash
python -m streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Using EchoMind

1. **Allow Camera Access**: Grant webcam permissions when prompted by your browser
2. **Select Dialect**: Choose between ASL or ISL from the sidebar
3. **Start Capturing**: Toggle "Run Capture" in the sidebar to begin gesture recognition
4. **Perform Signs**: Make sign language gestures in front of the camera
5. **Review Predictions**: Top candidate signs appear with confidence scores
6. **Confirm or Correct**: Click on the correct sign to improve future predictions
7. **Swipe to Undo**: Perform a right-to-left hand swipe to clear the current buffer

## 🔧 Configuration

### Sidebar Settings

- **Dialect**: Switch between ASL and ISL sign language variants
- **Run Capture**: Start/stop the camera feed and gesture detection
- **TTS Voice**: Configure the voice name for speech synthesis
- **Swipe Settings**: Adjust gesture detection thresholds
  - Dominant Hand: right, left, or auto-detect
  - Swipe dx threshold: Horizontal distance threshold
  - Swipe velocity threshold: Minimum speed for swipe detection
  - Swipe cooldown: Frames to wait between swipe detections

### Environment Configuration

The system uses environment variables for flexible deployment:

- **QDRANT_URL**: Your Qdrant Cloud cluster URL (auto-appends :6333 if missing)
- **QDRANT_API_KEY**: API key for Qdrant Cloud authentication
- **PIPER_MODEL**: Path to Piper ONNX model file
- **PIPER_CONFIG**: Path to Piper model configuration JSON
- **PIPER_USE_CUDA**: Enable GPU acceleration (true/false)

## 📁 Project Structure

```
EchoMind/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration (create this)
├── .gitignore                 # Git ignore rules
├── SPEC.md                    # Technical specification
├── README.md                  # This file
│
├── core/                      # Core application logic
│   ├── controller.py          # Gesture controller and state management
│   └── gesture_processor.py  # Motion processing and feature extraction
│
├── database/                  # Database and vector storage
│   └── qdrant_manager.py     # Qdrant Cloud client and operations
│
├── models/                    # TTS voice models
│   ├── en_US-amy-medium.onnx
│   └── en_US-amy-medium.onnx.json
│
├── cache/                     # Audio cache (auto-generated)
│   └── audio/
│
└── ts/                        # Third-party utilities
```

## 🧠 How It Works

### Architecture Overview

1. **Capture**: Webcam video is processed at ~30 FPS using MediaPipe Holistic
2. **Landmark Extraction**: Hand, pose, and facial landmarks are extracted and normalized
3. **Motion Encoding**: Temporal windows of landmarks are encoded into 256-dimensional vectors
4. **Vector Search**: Query vectors are matched against Qdrant Cloud database using cosine similarity
5. **Dialect Filtering**: Results are filtered by selected dialect (ASL/ISL)
6. **Speech Synthesis**: Recognized glosses are converted to speech using Piper TTS or pyttsx3
7. **Reinforcement Learning**: User corrections update the database to improve future predictions

### Key Components

- **MediaPipe Holistic**: Provides real-time hand, pose, and face landmark detection
- **Qdrant Cloud**: Vector database for fast similarity search with named vectors
- **Piper TTS**: High-quality neural text-to-speech with offline capability
- **Streamlit**: Interactive web interface with live video feed
- **DiskCache**: Persistent audio caching for faster playback

## 🔍 Troubleshooting

### Qdrant Connection Issues

**Problem**: "Failed to connect to Qdrant Cloud"

**Solutions**:
- Verify `QDRANT_URL` and `QDRANT_API_KEY` in `.env` file
- Ensure URL includes `:6333` port (auto-appended if missing)
- Check internet connectivity
- System automatically falls back to in-memory database if cloud unavailable

### Piper TTS Not Working

**Problem**: Audio not playing or using fallback voice

**Solutions**:
- Verify `PIPER_MODEL` and `PIPER_CONFIG` paths are correct
- Ensure model files exist in the `models/` directory
- System automatically falls back to pyttsx3 if Piper unavailable
- Download voices from [Piper Voices](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md)

### Camera Access Denied

**Problem**: "Unable to access webcam"

**Solutions**:
- Grant camera permissions in your browser
- Check if another application is using the webcam
- Verify webcam is properly connected
- Try restarting the browser or application

### Protobuf Version Conflicts

**Problem**: Import errors or compatibility issues

**Solutions**:
```bash
pip install --upgrade "protobuf>=4.25.3,<5"
```

### Slow Performance

**Solutions**:
- Reduce video resolution in MediaPipe settings
- Adjust gesture detection thresholds in sidebar
- Use GPU acceleration if available (`PIPER_USE_CUDA=true`)
- Ensure adequate system resources (4GB+ RAM recommended)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is part of the Convolve 4.0 "Accessibility & Inclusion" challenge.

## 🙏 Acknowledgments

- MediaPipe team for the holistic landmark detection model
- Qdrant for the vector database platform
- Piper TTS for offline neural speech synthesis
- The sign language community for inspiration and feedback

## 📞 Support

For questions or issues, please open an issue on the GitHub repository.
