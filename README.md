# 🎬 AI-Powered Screenplay & Transcript Metadata Extraction Platform

An enterprise-grade, NLP-driven Metadata Extraction and Screenplay Analytics platform built with **FastAPI**, **Streamlit**, **spaCy**, **Scikit-Learn**, and **SQLite**. 

The platform automatically analyzes raw screenplay scripts and timestamped subtitle transcripts (`.srt` files) to extract high-precision character pacing metrics, timeline dialogue streams, dynamic topics, named entities, sentiment scores, and genre classifications.

---

## 🌟 Key Features

- **⏱️ Character Presence & Screen Time Pacing**:
  - Classifies character roles dynamically into `Lead`, `Supporting`, or `Minor`.
  - Calculates accurate screen time percentage (`0.0% - 100.0%`).
  - Tracks first entry timestamp, last exit timestamp, scene counts, and total dialogue lines.

- **💬 Timeline Dialogue Breakdown Stream**:
  - Interactive, searchable stream of timestamped dialogues.
  - Filter dialogues by specific character/speaker or keywords.

- **🎯 Primary & Secondary Category Classification**:
  - Predicts screenplay genres (e.g., `Drama`, `Action`, `Comedy`, `Sci-Fi`, `War`) with confidence metrics and reasoning.

- **🏷️ Dynamic Topic & TF-IDF Keyword Extraction**:
  - Extracts main narrative themes, storyline subjects, and statistical TF-IDF key terms without hardcoded dictionaries.

- **🔍 Named Entity Recognition (NER)**:
  - Extracts entities categorized by **People/Characters**, **Locations & Setting**, **Organizations**, and **Products/Objects**.

- **🎭 Sentiment & Emotional Tone Analysis**:
  - Multi-dimensional sentiment scoring (`Positive`, `Negative`, `Neutral`, `Mixed`) with confidence metrics and emotional state tags.

- **⏳ Time-Window Duration Filtering**:
  - Analyze specific time segments of a movie (e.g., `00:10:00` to `00:30:00`) or the entire screenplay duration.

- **⚡ Instant SQLite Caching & Database Persistence**:
  - Automatic lookup from `data/transcript_metadata.db` for instant retrieval of previously analyzed scripts, with an optional force re-extract toggle.

- **📁 SRT Subtitle File Processing**:
  - Upload custom `.srt` subtitle transcript files for on-the-fly NLP extraction.

---

## 🏗️ Architecture & Project Structure

```
Metadata-Mapping/
├── api/
│   └── main.py                 # FastAPI REST Endpoints (/api/process, /api/movies, /api/process-srt, /api/metadata)
├── frontend/
│   └── app.py                  # Modern Glassmorphism Streamlit UI Dashboard
├── pipeline/
│   └── processor.py            # Core Metadata Extraction Pipeline Coordinator
├── extractors/
│   ├── base_extractor.py       # Abstract Extractor Base Class
│   ├── topic_extractor.py      # TF-IDF & Topic Extractor
│   ├── entity_extractor.py     # spaCy NER Entity Extractor
│   ├── sentiment_extractor.py  # Local NLP Sentiment Extractor
│   ├── category_extractor.py   # Machine Learning & Genre Similarity Classifier
│   └── character_presence_extractor.py # Character Pacing & Screen Time Calculator
├── data/
│   ├── dataset_loader.py       # Kaggle Movie Scripts Zip Archive Loader
│   ├── db_manager.py           # SQLite Database Manager (data/transcript_metadata.db)
│   └── timestamp_parser.py     # Timestamp estimation, filtering, & robust SRT parser
├── models/
│   └── schema.py               # Pydantic v2 Validation Schemas & Data Models
├── config/
│   └── config.yaml             # Application Configuration & Hyperparameters
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation
```

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/Shirish2103/Metadata-Mapping.git
cd Metadata-Mapping
```

### 3. Create & Activate a Virtual Environment (Recommended)

- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download spaCy Language Model
Download the English NLP model required for Named Entity Recognition (NER) and POS tagging:
```bash
python -m spacy download en_core_web_sm
```

---

## 🏃 Running the Application

To run the complete platform, start both the **FastAPI Backend** and **Streamlit Frontend** services.

### Step 1: Launch FastAPI Backend Server
Open a terminal in the project directory and run:
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
- **Backend API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger API Documentation**: `http://127.0.0.1:8000/docs`

---

### Step 2: Launch Streamlit Frontend Dashboard
Open a second terminal window in the project directory and run:
```bash
python -m streamlit run frontend/app.py --server.port 8501
```
- **Streamlit Web Dashboard**: `http://localhost:8501`

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/api/health` | Health check endpoint |
| **GET** | `/api/movies` | Returns list of available movie scripts from dataset archive |
| **POST** | `/api/process` | Analyzes script by Movie Title / IMDB ID with time window filters (checks SQLite cache first) |
| **POST** | `/api/process-srt` | Accepts uploaded `.srt` subtitle file for real-time metadata extraction |
| **GET** | `/api/metadata` | Directly queries stored metadata records from SQLite database |

### Example Request (`POST /api/process`):
```json
{
  "title_or_imdb": "Full Metal Jacket",
  "start_time": "00:10:00",
  "end_time": "00:30:00",
  "force_refresh": false
}
```

---

## 🧪 Testing & Verification

You can verify the backend endpoints using `curl` or Python `requests`:

```bash
# Test API Health
curl http://127.0.0.1:8000/api/health

# Process Movie Metadata
curl -X POST "http://127.0.0.1:8000/api/process" \
     -H "Content-Type: application/json" \
     -d '{"title_or_imdb": "Full Metal Jacket", "start_time": "00:05:00", "end_time": "00:25:00"}'
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
