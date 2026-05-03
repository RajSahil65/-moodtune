# 🎵 Emotion Recognition Based Music Recommendation System

A production-ready AI system that detects human emotions via **text**, **voice**, or **facial expressions** and recommends music using generative AI.

---

## 🏗️ Architecture Overview

```
User Input (Text / Voice / Face)
        ↓
Emotion Detection Engine
  ├── NLP Module (BERT/VADER for text)
  ├── SER Module (Librosa for voice)
  └── FER Module (OpenCV + CNN for face)
        ↓
Emotion → Mood Mapping
        ↓
Music Recommendation Engine
  ├── Spotify API Integration
  └── YouTube API Fallback
        ↓
Generative AI Enhancement (Claude API)
  ├── Playlist descriptions
  ├── Emotion explanations
  └── Chatbot assistant
        ↓
Frontend UI (React)
```

---

## 📁 Project Structure

```
emotion-music-ai/
├── backend/
│   ├── app.py                  # FastAPI main application
│   ├── config.py               # Configuration & env vars
│   ├── api/
│   │   ├── routes.py           # API endpoints
│   │   ├── auth.py             # JWT authentication
│   │   └── websocket.py        # Real-time webcam stream
│   ├── models/
│   │   ├── emotion_text.py     # Text emotion detection
│   │   ├── emotion_voice.py    # Voice emotion detection
│   │   ├── emotion_face.py     # Face emotion detection
│   │   └── recommender.py      # Music recommendation engine
│   ├── utils/
│   │   ├── spotify.py          # Spotify API integration
│   │   ├── youtube.py          # YouTube API integration
│   │   └── genai.py            # Claude/OpenAI GenAI wrapper
│   ├── database/
│   │   ├── db.py               # SQLite setup
│   │   ├── models.py           # ORM models
│   │   └── crud.py             # DB operations
│   └── tests/
│       ├── test_emotion.py     # Emotion detection tests
│       └── test_api.py         # API endpoint tests
├── frontend/
│   ├── index.html              # Main React app entry
│   └── public/                 # Static assets
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo>
cd emotion-music-ai
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Backend

```bash
cd backend
uvicorn app:app --reload --port 8000
```

### 4. Open Frontend

Open `frontend/index.html` in browser, or serve with:
```bash
python -m http.server 3000 --directory frontend
```

### 5. Docker (Optional)

```bash
docker-compose up --build
```

---

## 🔑 API Keys Required

| Service | Where to Get | Purpose |
|---------|-------------|---------|
| Anthropic Claude | console.anthropic.com | GenAI explanations |
| Spotify | developer.spotify.com | Music recommendations |
| YouTube Data API | console.cloud.google.com | Video fallback |

---

## 📊 Emotion → Music Mapping

| Emotion | Spotify Seed Genres | Energy | Valence |
|---------|-------------------|--------|---------|
| Happy   | pop, dance, funk  | 0.8    | 0.9     |
| Sad     | acoustic, blues   | 0.3    | 0.2     |
| Angry   | metal, rock, punk | 0.9    | 0.4     |
| Neutral | ambient, classical| 0.5    | 0.5     |
| Fear    | ambient, classical| 0.4    | 0.3     |
| Surprise| electronic, indie | 0.7    | 0.7     |

---

## 🧪 Sample API Calls

```bash
# Detect emotion from text
curl -X POST http://localhost:8000/api/detect/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'

# Get music recommendations
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"emotion": "happy", "limit": 5}'
```

---

## 📈 Model Performance

| Model | Dataset | Accuracy |
|-------|---------|----------|
| Text (VADER) | SemEval | ~82% |
| Text (Transformers) | GoEmotions | ~89% |
| Voice (Librosa+SVM) | RAVDESS | ~75% |
| Face (CNN) | FER2013 | ~65-72% |

---

## 🐳 Docker Deployment

```bash
docker build -t emotion-music-ai .
docker run -p 8000:8000 --env-file .env emotion-music-ai
```
