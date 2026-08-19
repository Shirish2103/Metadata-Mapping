import os
import re
import yaml
import concurrent.futures
from typing import List, Dict, Any, Optional
from models.schema import (
    MovieMetadata, SceneSegment, Dialogue, ScriptMetadataResult, TimeRange,
    ExtractedTopics, ExtractedEntities, ExtractedSentiment, ExtractedCategory, DialogueEntry,
    CharacterPresenceReport
)
from data.dataset_loader import DatasetLoader
from data.db_manager import DatabaseManager
from data.timestamp_parser import TimestampParser, timestamp_to_seconds, format_seconds_to_timestamp
from extractors.topic_extractor import TopicExtractor
from extractors.entity_extractor import EntityExtractor
from extractors.sentiment_extractor import SentimentExtractor
from extractors.category_extractor import CategoryExtractor
from extractors.character_presence_extractor import CharacterPresenceExtractor

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
        self.character_presence_extractor = CharacterPresenceExtractor(self.config)


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
        movie_info = self.loader.load_movie_metadata(imdb_id_or_title)
        if not movie_info:
            raise ValueError(f"Movie metadata not found for query: '{imdb_id_or_title}'")

        scenes = self.loader.load_screenplay_scenes(movie_info.imdb_id)
        if not scenes:
            # Generate movie-specific scenes & dialogues using real cast and plot info
            cast_members = movie_info.cast if (movie_info.cast and len(movie_info.cast) > 0) else ["Protagonist", "Supporting Character", "Narrator"]
            plot_text = movie_info.plot if movie_info.plot else f"Story of {movie_info.title}."
            
            plot_parts = [p.strip() for p in re.split(r'[.!?]+', plot_text) if p.strip()]
            if not plot_parts or len(plot_parts) < 3:
                plot_parts = [
                    f"Opening scene introducing {movie_info.title}.",
                    f"Main central narrative conflict unfolds.",
                    f"{plot_text}",
                    f"Climax and emotional resolution for {movie_info.title}."
                ]

            genre_loc = movie_info.genres[0].upper() if (movie_info.genres and len(movie_info.genres) > 0) else "DRAMA"
            scenes = []
            time_per_scene = 900.0  # 15 minutes per scene

            for s_idx, part in enumerate(plot_parts):
                speaker = cast_members[s_idx % len(cast_members)].title()
                start_sec = s_idx * time_per_scene
                end_sec = (s_idx + 1) * time_per_scene
                
                dlg = Dialogue(
                    speaker=speaker,
                    text=part,
                    scene_idx=s_idx + 1,
                    time_range=TimeRange(
                        start_time=format_seconds_to_timestamp(start_sec),
                        end_time=format_seconds_to_timestamp(end_sec),
                        is_estimated=True
                    )
                )
                
                scenes.append(
                    SceneSegment(
                        scene_idx=s_idx + 1,
                        location=f"SCENE {s_idx + 1} - {genre_loc}",
                        dialogues=[dlg],
                        action_text=f"Narrative scene sequence for {movie_info.title}.",
                        time_range=TimeRange(
                            start_time=format_seconds_to_timestamp(start_sec),
                            end_time=format_seconds_to_timestamp(end_sec),
                            is_estimated=True
                        )
                    )
                )

        self.db.save_scenes(movie_info.imdb_id, scenes)

        full_movie_start = scenes[0].time_range.start_time if (scenes and scenes[0].time_range) else "00:00:00"
        full_movie_end = scenes[-1].time_range.end_time if (scenes and scenes[-1].time_range) else "00:00:00"
        full_dur_sec = max(0.0, timestamp_to_seconds(full_movie_end) - timestamp_to_seconds(full_movie_start))
        full_total_duration_str = format_seconds_to_timestamp(full_dur_sec)

        is_windowed = False
        if start_time or end_time:
            s_ts = start_time if start_time else "00:00:00"
            e_ts = end_time if end_time else (scenes[-1].time_range.end_time if scenes and scenes[-1].time_range else "99:59:59")
            scenes = TimestampParser.filter_scenes_by_timerange(scenes, s_ts, e_ts)
            is_windowed = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_topics = executor.submit(self.topic_extractor.extract, movie_info, scenes)
            f_entities = executor.submit(self.entity_extractor.extract, movie_info, scenes)
            f_sentiment = executor.submit(self.sentiment_extractor.extract, movie_info, scenes)
            f_category = executor.submit(self.category_extractor.extract, movie_info, scenes)
            f_char = executor.submit(self.character_presence_extractor.extract, movie_info, scenes)

            topics: ExtractedTopics = f_topics.result()
            entities: ExtractedEntities = f_entities.result()
            sentiment: ExtractedSentiment = f_sentiment.result()
            category: ExtractedCategory = f_category.result()
            character_presence: CharacterPresenceReport = f_char.result()


        if scenes and scenes[0].time_range and scenes[-1].time_range:
            actual_start = start_time if start_time else scenes[0].time_range.start_time
            actual_end = end_time if end_time else scenes[-1].time_range.end_time
        else:
            actual_start = start_time if start_time else "00:00:00"
            actual_end = end_time if end_time else "00:00:00"

        overall_time_range = TimeRange(
            start_time=actual_start,
            end_time=actual_end,
            total_duration=full_total_duration_str,
            is_estimated=True
        )

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

        speakers = set()
        dialogues_in_window = []
        for sc in scenes:
            scene_ts = sc.time_range.start_time if sc.time_range else "00:00:00"
            for d in sc.dialogues:
                if d.speaker:
                    speakers.add(d.speaker.title())
                    dialogues_in_window.append(
                        DialogueEntry(
                            timestamp=d.time_range.start_time if d.time_range else scene_ts,
                            speaker=d.speaker.title(),
                            text=d.text,
                            location=sc.location
                        )
                    )

        result = ScriptMetadataResult(
            imdb_id=movie_info.imdb_id,
            title=movie_info.title + (f" [{actual_start} - {actual_end}]" if is_windowed else ""),
            movie_info=movie_info,
            time_range=overall_time_range,
            total_duration=full_total_duration_str,
            topics=topics,
            entities=entities,
            sentiment=sentiment,
            category=category,
            scene_breakdown_count=len(scenes),
            speaker_list=sorted(list(speakers))[:20],
            dialogues_in_window=dialogues_in_window,
            character_presence=character_presence
        )


        self.db.save_extracted_metadata(result)

        return result

    def process_srt_file(
        self, 
        filepath: str, 
        title: str = "External SRT Transcript",
        start_time: Optional[str] = None, 
        end_time: Optional[str] = None
    ) -> ScriptMetadataResult:
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

        full_srt_start = srt_blocks[0].get("start_time", "00:00:00")
        full_srt_end = srt_blocks[-1].get("end_time", "00:00:00")
        full_dur_sec = max(0.0, timestamp_to_seconds(full_srt_end) - timestamp_to_seconds(full_srt_start))
        full_total_duration_str = format_seconds_to_timestamp(full_dur_sec)

        overall_tr = TimeRange(
            start_time=start_time if start_time else full_srt_start,
            end_time=end_time if end_time else full_srt_end,
            total_duration=full_total_duration_str,
            is_estimated=False
        )

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
            e_ts = end_time if end_time else full_srt_end
            scenes = TimestampParser.filter_scenes_by_timerange(scenes, s_ts, e_ts)
            is_windowed = True
        plot_summary = f"External timestamped transcript [{overall_tr.start_time} - {overall_tr.end_time}]" if is_windowed else "External timestamped transcript"

        movie_info = MovieMetadata(
            imdb_id=imdb_id,
            title=title,
            year="2026",
            genres=["Entertainment"],
            plot=plot_summary
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_topics = executor.submit(self.topic_extractor.extract, movie_info, scenes)
            f_entities = executor.submit(self.entity_extractor.extract, movie_info, scenes)
            f_sentiment = executor.submit(self.sentiment_extractor.extract, movie_info, scenes)
            f_category = executor.submit(self.category_extractor.extract, movie_info, scenes)
            f_char = executor.submit(self.character_presence_extractor.extract, movie_info, scenes)

            topics = f_topics.result()
            entities = f_entities.result()
            sentiment = f_sentiment.result()
            category = f_category.result()
            character_presence = f_char.result()


        dialogues_in_window = []
        speakers = set()
        for sc in scenes:
            scene_ts = sc.time_range.start_time if sc.time_range else "00:00:00"
            for d in sc.dialogues:
                if d.speaker:
                    speakers.add(d.speaker.title())
                    dialogues_in_window.append(
                        DialogueEntry(
                            timestamp=d.time_range.start_time if d.time_range else scene_ts,
                            speaker=d.speaker.title(),
                            text=d.text,
                            location=sc.location
                        )
                    )

        result = ScriptMetadataResult(
            imdb_id=imdb_id,
            title=title + (f" [{overall_tr.start_time} - {overall_tr.end_time}]" if is_windowed else ""),
            movie_info=movie_info,
            time_range=overall_tr,
            total_duration=full_total_duration_str,
            topics=topics,
            entities=entities,
            sentiment=sentiment,
            category=category,
            scene_breakdown_count=len(scenes),
            speaker_list=sorted(list(speakers)),
            dialogues_in_window=dialogues_in_window,
            character_presence=character_presence
        )


        self.db.save_extracted_metadata(result)
        return result
