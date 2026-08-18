import os
import pytest
from models.schema import (
    TimeRange, MovieMetadata, SceneSegment, Dialogue,
    ExtractedTopics, ExtractedEntities, ExtractedSentiment, ExtractedCategory
)
from data.timestamp_parser import TimestampParser, format_seconds_to_timestamp
from data.dataset_loader import DatasetLoader
from data.db_manager import DatabaseManager
from pipeline.processor import MetadataProcessor

def test_schema_models():
    tr = TimeRange(start_time="00:01:00", end_time="00:03:00", is_estimated=True)
    assert tr.start_time == "00:01:00"
    assert tr.is_estimated is True

    movie = MovieMetadata(imdb_id="1179933", title="10 Cloverfield Lane", year="2016", genres=["Thriller", "Drama"])
    assert movie.imdb_id == "1179933"
    assert "Thriller" in movie.genres

def test_timestamp_parser():
    ts = format_seconds_to_timestamp(125.5)
    assert ts == "00:02:05"

    srt_data = """1
00:00:10,000 --> 00:00:15,000
Tony: We have a problem.

2
00:00:16,000 --> 00:00:20,000
Steve: What happened?
"""
    parsed = TimestampParser.parse_srt(srt_data)
    assert len(parsed) == 2
    assert parsed[0]["speaker"] == "Tony"
    assert parsed[0]["text"] == "We have a problem."
    assert parsed[1]["speaker"] == "Steve"

def test_timestamp_to_seconds_and_filtering():
    from data.timestamp_parser import timestamp_to_seconds
    assert timestamp_to_seconds("00:10:00") == 600.0
    assert timestamp_to_seconds("10:00") == 600.0
    assert timestamp_to_seconds("120") == 120.0

    scene1 = SceneSegment(
        scene_idx=1,
        location="ROOM 1",
        time_range=TimeRange(start_time="00:00:00", end_time="00:05:00"),
        dialogues=[
            Dialogue(speaker="Tony", text="Hello", scene_idx=1, time_range=TimeRange(start_time="00:01:00", end_time="00:02:00"))
        ]
    )
    scene2 = SceneSegment(
        scene_idx=2,
        location="ROOM 2",
        time_range=TimeRange(start_time="00:10:00", end_time="00:15:00"),
        dialogues=[
            Dialogue(speaker="Steve", text="World", scene_idx=2, time_range=TimeRange(start_time="00:11:00", end_time="00:12:00"))
        ]
    )

    filtered = TimestampParser.filter_scenes_by_timerange([scene1, scene2], "00:08:00", "00:20:00")
    assert len(filtered) == 1
    assert filtered[0].scene_idx == 2
    assert len(filtered[0].dialogues) == 1

def test_dataset_loader():
    archive_path = r"d:\Downloads\archive.zip"
    if not os.path.exists(archive_path):
        pytest.skip("Dataset archive not found")

    loader = DatasetLoader(archive_path)
    movies = loader.get_available_movies()
    assert len(movies) > 0

    meta = loader.load_movie_metadata("1179933")
    assert meta is not None
    assert meta.title == "10 Cloverfield Lane"

    scenes = loader.load_screenplay_scenes("1179933")
    assert len(scenes) > 0
    assert scenes[0].time_range is not None
    assert scenes[0].time_range.start_time == "00:00:00"

def test_database_manager(tmp_path):
    db_file = str(tmp_path / "test_metadata.db")
    db = DatabaseManager(db_path=db_file)

    meta = MovieMetadata(imdb_id="test1234", title="Test Movie", year="2025")
    db.save_movie(meta)

    retrieved = db.get_metadata("test1234")
    assert retrieved is None  # Extracted metadata not yet saved

def test_end_to_end_processor():
    archive_path = r"d:\Downloads\archive.zip"
    if not os.path.exists(archive_path):
        pytest.skip("Dataset archive not found")

    processor = MetadataProcessor()
    result = processor.process_transcript("10 Cloverfield Lane_1179933")

    assert result.imdb_id == "1179933"
    assert "10 Cloverfield Lane" in result.title
    assert len(result.topics.keywords) > 0
    assert result.category.primary_category in ["Thriller", "Drama", "Action", "Entertainment"]
    assert result.sentiment.sentiment in ["Positive", "Negative", "Neutral", "Mixed"]
    assert len(result.speaker_list) > 0

def test_time_window_processor():
    archive_path = r"d:\Downloads\archive.zip"
    if not os.path.exists(archive_path):
        pytest.skip("Dataset archive not found")

    processor = MetadataProcessor()
    result = processor.process_transcript("1179933", start_time="00:05:00", end_time="00:20:00")

    assert result.imdb_id == "1179933"
    assert "00:05:00" in result.title or result.time_range.start_time == "00:05:00"
    assert result.time_range.start_time == "00:05:00"
    assert result.time_range.end_time == "00:20:00"

