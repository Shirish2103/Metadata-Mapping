import os
import tempfile
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.processor import MetadataProcessor
from data.dataset_loader import DatasetLoader
from data.db_manager import DatabaseManager
from models.schema import ScriptMetadataResult

app = FastAPI(
    title="AI Transcript Metadata Extraction API",
    description="FastAPI Backend for dynamic screenplay metadata extraction, character presence analysis, timeline dialogue breakdown, and sentiment scoring.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared components
processor = MetadataProcessor()
loader = DatasetLoader()
db = DatabaseManager()


class ProcessRequest(BaseModel):
    title_or_imdb: str = Field(..., description="Movie Title or IMDB ID (e.g., 'Full Metal Jacket' or 'tt0093058')")
    start_time: Optional[str] = Field(None, description="Start timestamp filter e.g., '00:10:00'")
    end_time: Optional[str] = Field(None, description="End timestamp filter e.g., '00:30:00'")
    force_refresh: Optional[bool] = Field(False, description="Set True to bypass database cache and re-extract metadata")


import threading


def _background_cache_warmup():
    try:
        from scripts.preindex_dataset import preindex_all_dataset_movies
        print("🚀 Starting background dataset metadata cache warmup...")
        preindex_all_dataset_movies(limit=15)
    except Exception as e:
        print(f"Background warmup notice: {e}")


@app.on_event("startup")
def startup_event():
    # Start non-blocking background worker thread to warm up SQLite cache
    thread = threading.Thread(target=_background_cache_warmup, daemon=True)
    thread.start()


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "AI Transcript Metadata Extraction Backend",
        "version": "1.0.0"
    }


@app.get("/api/movies", response_model=List[Dict[str, Any]])
def list_available_movies():
    """Returns a list of available movie scripts in the dataset archive."""
    try:
        movies = loader.get_available_movies()
        return movies
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list movies: {str(e)}")


@app.post("/api/process", response_model=Dict[str, Any])
def process_transcript_metadata(req: ProcessRequest):
    """
    Processes a screenplay by Movie Title or IMDB ID with optional Start Time and End Time filters.
    Checks SQLite database first. If present, returns saved data; otherwise runs dynamic NLP extraction and saves to DB.
    """
    if not req.title_or_imdb or not req.title_or_imdb.strip():
        raise HTTPException(status_code=400, detail="Movie title or IMDB ID is required.")

    target = req.title_or_imdb.strip()
    start_ts = req.start_time.strip() if req.start_time else None
    end_ts = req.end_time.strip() if req.end_time else None

    try:
        # 1. Check if metadata is already cached/stored in SQLite Database unless force_refresh is requested
        if not req.force_refresh:
            stored_record = db.get_metadata(target, start_time=start_ts, end_time=end_ts)
            if stored_record:
                stored_record["fetch_source"] = "database"
                return stored_record

        # 2. Otherwise run pipeline extraction and save to database
        result: ScriptMetadataResult = processor.process_transcript(
            imdb_id_or_title=target,
            start_time=start_ts,
            end_time=end_ts
        )
        res_dict = result.model_dump()
        res_dict["fetch_source"] = "extracted_and_saved"
        return res_dict
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata extraction error: {str(e)}")


@app.post("/api/process-srt", response_model=Dict[str, Any])
def process_srt_file(
    file: UploadFile = File(...),
    start_time: Optional[str] = Form(None),
    end_time: Optional[str] = Form(None)
):
    """
    Processes an uploaded SRT subtitle transcript file with optional Start Time and End Time.
    """
    if not file.filename.endswith(('.srt', '.txt')):
        raise HTTPException(status_code=400, detail="File must be an SRT (.srt) subtitle transcript.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp:
            contents = file.file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        clean_title = os.path.splitext(file.filename)[0].replace('.', ' ').strip()
        result: ScriptMetadataResult = processor.process_srt_file(
            filepath=tmp_path,
            title=clean_title,
            start_time=start_time.strip() if start_time else None,
            end_time=end_time.strip() if end_time else None
        )

        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SRT processing failed: {str(e)}")


@app.get("/api/metadata", response_model=Dict[str, Any])
def query_stored_metadata(
    query: str = Query(..., description="Movie Title or IMDB ID to search in database"),
    start_time: Optional[str] = Query(None, description="Start timestamp window"),
    end_time: Optional[str] = Query(None, description="End timestamp window")
):
    """
    Retrieves stored extracted metadata from SQLite database by movie title or IMDB ID.
    """
    record = db.get_metadata(
        imdb_id_or_title=query.strip(),
        start_time=start_time.strip() if start_time else None,
        end_time=end_time.strip() if end_time else None
    )
    if not record:
        raise HTTPException(
            status_code=404, 
            detail=f"No stored metadata found for query '{query}'. Try running /api/process first!"
        )
    return record
