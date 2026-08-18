import os
import yaml
from typing import List, Dict, Any, Optional
from models.schema import (
    MovieMetadata, SceneSegment, Dialogue, ScriptMetadataResult, TimeRange,
    ExtractedTopics, ExtractedEntities, ExtractedSentiment, ExtractedCategory
)
from data.dataset_loader import DatasetLoader
from data.db_manager import DatabaseManager
from data.timestamp_parser import TimestampParser
from extractors.topic_extractor import TopicExtractor
from extractors.entity_extractor import EntityExtractor
from extractors.sentiment_extractor import SentimentExtractor
from extractors.category_extractor import CategoryExtractor

class MetadataProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        archive_path = self.config.get("dataset", {}).get("archive_path", r"d:\Downloads\archive.zip")
        db_path = self.config.get("database", {}).get("db_path", "data/transcript_metadata.db")

        self.loader = DatasetLoader(archive_path=archive_path)
        self.db = DatabaseManager(db_path=db_path)

        self.topic_extractor = TopicExtractor(self.config)
        self.entity_extractor = EntityExtractor(self.config)
        self.sentiment_extractor = SentimentExtractor(self.config)
        self.category_extractor = CategoryExtractor(self.config)

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def process_transcript(
        self, 
        imdb_id_or_title: str, 
        start_time: Optional[str] = None, 
        end_time: Optional[str] = None
    ) -> ScriptMetadataResult:
        """Processes a movie script transcript from the Kaggle dataset."""
        # 1. Load Movie Metadata
        movie_info = self.loader.load_movie_metadata(imdb_id_or_title)
        if not movie_info:
            raise ValueError(f"Movie metadata not found for query: '{imdb_id_or_title}'")

        # 2. Load Screenplay Scenes & Estimate Timestamps
        scenes = self.loader.load_screenplay_scenes(movie_info.imdb_id)
        if not scenes:
            raise ValueError(f"Screenplay JSON transcript not found for: '{movie_info.title}' ({movie_info.imdb_id})")

        # 3. Save full scenes and dialogues to database
        self.db.save_scenes(movie_info.imdb_id, scenes)

        # Filter scenes by time window if start_time or end_time is provided
        is_windowed = False
        if start_time or end_time:
            s_ts = start_time if start_time else "00:00:00"
            e_ts = end_time if end_time else (scenes[-1].time_range.end_time if scenes and scenes[-1].time_range else "99:59:59")
            scenes = TimestampParser.filter_scenes_by_timerange(scenes, s_ts, e_ts)
            is_windowed = True

        # 4. Extract Metadata Categories
        topics: ExtractedTopics = self.topic_extractor.extract(movie_info, scenes)
        entities: ExtractedEntities = self.entity_extractor.extract(movie_info, scenes)
        sentiment: ExtractedSentiment = self.sentiment_extractor.extract(movie_info, scenes)
        category: ExtractedCategory = self.category_extractor.extract(movie_info, scenes)

        # 5. Calculate time range
        if scenes and scenes[0].time_range and scenes[-1].time_range:
            actual_start = start_time if start_time else scenes[0].time_range.start_time
            actual_end = end_time if end_time else scenes[-1].time_range.end_time
        else:
            actual_start = start_time if start_time else "00:00:00"
            actual_end = end_time if end_time else "00:00:00"

        overall_time_range = TimeRange(start_time=actual_start, end_time=actual_end, is_estimated=True)

        if is_windowed:
            movie_info = movie_info.model_copy(deep=True)
            if scenes:
                actions = []
                for sc in scenes:
                    if sc.action_text:
                        line = sc.action_text.strip().splitlines()[0]
                        if line:
                            actions.append(line)
                    elif sc.dialogues:
                        line = f"{sc.dialogues[0].speaker}: {sc.dialogues[0].text}"
                        actions.append(line)
                if actions:
                    movie_info.plot = f"Time Window [{actual_start} - {actual_end}] Plot: " + " | ".join(actions[:5])
                else:
                    movie_info.plot = f"Scenes occurring between {actual_start} and {actual_end}."
            else:
                movie_info.plot = f"No scene activity found between {actual_start} and {actual_end}."

        # Collect unique speakers
        speakers = set()
        for sc in scenes:
            for d in sc.dialogues:
                if d.speaker:
                    speakers.add(d.speaker.title())

        # 6. Aggregate into Pydantic ScriptMetadataResult
        result = ScriptMetadataResult(
            imdb_id=movie_info.imdb_id,
            title=movie_info.title + (f" [{actual_start} - {actual_end}]" if is_windowed else ""),
            movie_info=movie_info,
            time_range=overall_time_range,
            topics=topics,
            entities=entities,
            sentiment=sentiment,
            category=category,
            scene_breakdown_count=len(scenes),
            speaker_list=sorted(list(speakers))[:20]
        )

        # 7. Persist to SQLite
        self.db.save_extracted_metadata(result)

        return result

    def process_srt_file(
        self, 
        filepath: str, 
        title: str = "External SRT Transcript",
        start_time: Optional[str] = None, 
        end_time: Optional[str] = None
    ) -> ScriptMetadataResult:
        """Processes an external SRT transcript file with exact timestamps."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"SRT file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        srt_blocks = TimestampParser.parse_srt(content)
        if not srt_blocks:
            raise ValueError(f"Failed to parse any valid SRT blocks from {filepath}")

        if title == "External SRT Transcript":
            filename = os.path.basename(filepath)
            clean_title = os.path.splitext(filename)[0]
            title = clean_title.replace('.', ' ').strip()

        imdb_id = "SRT_" + str(abs(hash(filepath)) % 1000000)

        # Group SRT blocks into a single scene segment
        dialogues = []
        speakers = set()
        for idx, block in enumerate(srt_blocks):
            spk = block.get("speaker", "Unknown")
            speakers.add(spk)
            dialogues.append(
                Dialogue(
                    speaker=spk,
                    text=block.get("text", ""),
                    scene_idx=1,
                    time_range=TimeRange(
                        start_time=block.get("start_time", "00:00:00"),
                        end_time=block.get("end_time", "00:00:00"),
                        is_estimated=False
                    )
                )
            )

        start_ts = srt_blocks[0].get("start_time", "00:00:00")
        end_ts = srt_blocks[-1].get("end_time", "00:00:00")
        overall_tr = TimeRange(start_time=start_ts, end_time=end_ts, is_estimated=False)

        scene = SceneSegment(
            scene_idx=1,
            location="TRANSCRIPT",
            dialogues=dialogues,
            time_range=overall_tr
        )

        scenes = [scene]

        is_windowed = False
        if start_time or end_time:
            s_ts = start_time if start_time else "00:00:00"
            e_ts = end_time if end_time else end_ts
            scenes = TimestampParser.filter_scenes_by_timerange(scenes, s_ts, e_ts)
            is_windowed = True
            overall_tr = TimeRange(start_time=s_ts, end_time=e_ts, is_estimated=False)
            speakers = set()
            for sc in scenes:
                for d in sc.dialogues:
                    if d.speaker:
                        speakers.add(d.speaker)

        plot_summary = f"External timestamped transcript [{overall_tr.start_time} - {overall_tr.end_time}]" if is_windowed else "External timestamped transcript"

        movie_info = MovieMetadata(
            imdb_id=imdb_id,
            title=title,
            year="2026",
            genres=["Entertainment"],
            plot=plot_summary
        )

        topics = self.topic_extractor.extract(movie_info, scenes)
        entities = self.entity_extractor.extract(movie_info, scenes)
        sentiment = self.sentiment_extractor.extract(movie_info, scenes)
        category = self.category_extractor.extract(movie_info, scenes)

        result = ScriptMetadataResult(
            imdb_id=imdb_id,
            title=title + (f" [{overall_tr.start_time} - {overall_tr.end_time}]" if is_windowed else ""),
            movie_info=movie_info,
            time_range=overall_tr,
            topics=topics,
            entities=entities,
            sentiment=sentiment,
            category=category,
            scene_breakdown_count=len(scenes),
            speaker_list=sorted(list(speakers))
        )

        self.db.save_extracted_metadata(result)
        return result

