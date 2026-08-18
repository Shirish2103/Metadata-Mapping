import os
import zipfile
import csv
import json
import io
import re
from typing import List, Dict, Any, Optional
from models.schema import MovieMetadata, SceneSegment, Dialogue, TimeRange
from data.timestamp_parser import TimestampParser

class DatasetLoader:
    def __init__(self, archive_path: str = r"d:\Downloads\archive.zip"):
        self.archive_path = archive_path
        self._movie_meta_cache: Dict[str, MovieMetadata] = {}
        self._filename_map: Dict[str, str] = {}
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def _is_valid_speaker_nlp(self, speaker_candidate: str) -> Optional[str]:
        if not speaker_candidate:
            return None
        cand = re.sub(r'^[-*\s()"\':;.]+|[-*\s()"\':;.]+$', '', str(speaker_candidate)).strip()
        cand = re.sub(r'\(.*?\)', '', cand).strip()
        
        if not cand or len(cand) <= 1 or len(cand.split()) > 3:
            return None
            
        if self.nlp:
            doc = self.nlp(cand)
            for token in doc:
                if token.pos_ in {"NUM", "VERB", "PUNCT", "AUX", "DET", "ADP", "SCONJ", "CCONJ", "SYM"}:
                    return None
                if token.like_num or token.is_punct or token.is_space:
                    return None
            for ent in doc.ents:
                if ent.label_ in {"DATE", "TIME", "CARDINAL", "MONEY", "QUANTITY", "PERCENT", "ORDINAL"}:
                    return None
        else:
            if re.search(r'\d+', cand) or any(c in cand for c in ['"', '...', '!', '?', ';']):
                return None
                
        return cand.title()

    def _ensure_zip_open(self):
        if not os.path.exists(self.archive_path):
            raise FileNotFoundError(f"Archive zip not found at {self.archive_path}")
        return zipfile.ZipFile(self.archive_path, 'r')

    def get_available_movies(self) -> List[Dict[str, str]]:
        movies = []
        with self._ensure_zip_open() as zf:
            with zf.open('movie_metadata/movie_meta_data.csv') as f:
                content = f.read().decode('utf-8', errors='ignore')
                reader = csv.reader(io.StringIO(content))
                header = next(reader)
                for row in reader:
                    if row and len(row) >= 2:
                        imdb_id = row[0].strip()
                        title = row[1].strip()
                        year = row[3].strip() if len(row) > 3 else ""
                        genres = [g.strip() for g in row[22].split(',')] if len(row) > 22 and row[22] else []
                        movies.append({
                            "imdb_id": imdb_id,
                            "title": title,
                            "year": year,
                            "genres": ", ".join(genres)
                        })
        return movies

    def load_movie_metadata(self, imdb_id_or_title: str) -> Optional[MovieMetadata]:
        target = imdb_id_or_title.strip().lower()
        target_id = target.rsplit('_', 1)[-1] if '_' in target else target

        with self._ensure_zip_open() as zf:
            with zf.open('movie_metadata/movie_meta_data.csv') as f:
                content = f.read().decode('utf-8', errors='ignore')
                reader = csv.reader(io.StringIO(content))
                header = next(reader)
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    imdb_id = row[0].strip()
                    title = row[1].strip()
                    if (
                        imdb_id.lower() == target 
                        or imdb_id.lower() == target_id 
                        or title.lower() == target 
                        or target in title.lower()
                    ):
                        genres = [g.strip() for g in row[22].split(',')] if len(row) > 22 and row[22] else []
                        directors = [d.strip() for d in row[14].split(',')] if len(row) > 14 and row[14] else []
                        writers = [w.strip() for w in row[13].split(',')] if len(row) > 13 and row[13] else []
                        cast = [c.strip() for c in row[16].split(',')] if len(row) > 16 and row[16] else []
                        plot = row[19].strip() if len(row) > 19 else ""

                        return MovieMetadata(
                            imdb_id=imdb_id,
                            title=title,
                            year=row[3].strip() if len(row) > 3 else None,
                            genres=genres,
                            directors=directors,
                            writers=writers,
                            cast=cast,
                            plot=plot
                        )
        return None

    def load_screenplay_scenes(self, imdb_id_or_title: str) -> List[SceneSegment]:
        meta = self.load_movie_metadata(imdb_id_or_title)
        target_id = meta.imdb_id if meta else imdb_id_or_title
        raw_id = target_id.lower().replace('tt', '').strip()
        padded_id = raw_id.zfill(7)

        with self._ensure_zip_open() as zf:
            namelist = zf.namelist()
            matched_file = None
            
            annot_files = [
                f for f in namelist 
                if 'screenplay_data/data/rule_based_annotations/' in f and f.endswith('.json')
            ]
            if not annot_files:
                annot_files = [f for f in namelist if 'rule_based_annotations' in f and f.endswith('.json')]

            for f in annot_files:
                basename = os.path.basename(f)
                if f"_{padded_id}.json" in basename or f"_{raw_id}.json" in basename:
                    matched_file = f
                    break

            if not matched_file and meta:
                clean_target_title = re.sub(r'[^a-zA-Z0-9]', '', meta.title).lower()
                for f in annot_files:
                    basename = os.path.basename(f)
                    clean_file_title = re.sub(r'[^a-zA-Z0-9]', '', basename.rsplit('_', 1)[0]).lower()
                    if clean_target_title and (clean_target_title in clean_file_title or clean_file_title in clean_target_title):
                        matched_file = f
                        break

            if not matched_file:
                return []

            with zf.open(matched_file) as f:
                raw_json = json.loads(f.read().decode('utf-8', errors='ignore'))

            scenes: List[SceneSegment] = []
            scene_idx = 1

            for scene_data in raw_json:
                if not isinstance(scene_data, list):
                    continue

                location = "SCENE " + str(scene_idx)
                interior_exterior = None
                time_of_day = None
                dialogues = []
                action_text_lines = []

                for element in scene_data:
                    if not isinstance(element, dict):
                        continue
                    
                    head_type = element.get("head_type")
                    head_text = element.get("head_text", {})
                    text = element.get("text", "").strip()

                    if isinstance(head_text, dict):
                        if head_text.get("terior"):
                            interior_exterior = head_text.get("terior")
                        if head_text.get("location"):
                            loc_val = head_text.get("location")
                            if isinstance(loc_val, list):
                                items = []
                                for item in loc_val:
                                    if isinstance(item, list):
                                        items.append(" ".join(str(i) for i in item))
                                    else:
                                        items.append(str(item))
                                location = " ".join(items)
                            elif isinstance(loc_val, str):
                                location = loc_val
                        if head_text.get("ToD"):
                            time_of_day = head_text.get("ToD")

                    raw_speaker = None
                    if isinstance(head_text, dict):
                        if head_text.get("speaker/title"):
                            raw_speaker = str(head_text.get("speaker/title")).strip()
                        elif head_text.get("subj"):
                            raw_speaker = str(head_text.get("subj")).strip()

                    speaker = self._is_valid_speaker_nlp(raw_speaker)

                    if (head_type == "speaker/title" and speaker) or speaker:
                        if text:
                            dialogues.append(Dialogue(
                                speaker=speaker if speaker else "Unknown",
                                text=text,
                                scene_idx=scene_idx
                            ))
                    else:
                        if text:
                            action_text_lines.append(text)

                scene_segment = SceneSegment(
                    scene_idx=scene_idx,
                    location=location,
                    interior_exterior=interior_exterior,
                    time_of_day=time_of_day,
                    dialogues=dialogues,
                    action_text="\n".join(action_text_lines)
                )
                scenes.append(scene_segment)
                scene_idx += 1

            scenes = TimestampParser.estimate_screenplay_timestamps(scenes)
            return scenes
