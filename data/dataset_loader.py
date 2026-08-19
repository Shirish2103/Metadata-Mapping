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
    def __init__(self, dataset_dir: str = r"data\dataset", archive_path: str = r"d:\Downloads\archive.zip"):
        self.dataset_dir = dataset_dir
        self.archive_path = archive_path
        self._movie_meta_cache: Dict[str, MovieMetadata] = {}
        self._filename_map: Dict[str, str] = {}
        self._speaker_cache: Dict[str, Optional[str]] = {}
        self._csv_content_cache: Optional[str] = None
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def _is_valid_speaker_nlp(self, speaker_candidate: str) -> Optional[str]:
        if not speaker_candidate:
            return None
        if speaker_candidate in self._speaker_cache:
            return self._speaker_cache[speaker_candidate]

        cand = re.sub(r'^[-*\s()"\':;.]+|[-*\s()"\':;.]+$', '', str(speaker_candidate)).strip()
        cand = re.sub(r'\(.*?\)', '', cand).strip()
        
        if not cand or len(cand) <= 1 or len(cand.split()) > 3:
            self._speaker_cache[speaker_candidate] = None
            return None
            
        if self.nlp:
            doc = self.nlp(cand)
            for token in doc:
                if token.pos_ in {"NUM", "VERB", "PUNCT", "AUX", "DET", "ADP", "SCONJ", "CCONJ", "SYM"}:
                    self._speaker_cache[speaker_candidate] = None
                    return None
                if token.like_num or token.is_punct or token.is_space:
                    self._speaker_cache[speaker_candidate] = None
                    return None
            for ent in doc.ents:
                if ent.label_ in {"DATE", "TIME", "CARDINAL", "MONEY", "QUANTITY", "PERCENT", "ORDINAL"}:
                    self._speaker_cache[speaker_candidate] = None
                    return None
        else:
            if re.search(r'\d+', cand) or any(c in cand for c in ['"', '...', '!', '?', ';']):
                self._speaker_cache[speaker_candidate] = None
                return None
                
        res = cand.title()
        self._speaker_cache[speaker_candidate] = res
        return res

    def _get_metadata_csv_content(self) -> str:
        if self._csv_content_cache is not None:
            return self._csv_content_cache

        # 1. Try local extracted directory first
        local_csv = os.path.join(self.dataset_dir, "movie_metadata", "movie_meta_data.csv")
        if os.path.exists(local_csv):
            with open(local_csv, "r", encoding="utf-8", errors="ignore") as f:
                self._csv_content_cache = f.read()
                return self._csv_content_cache

        # 2. Fallback to archive zip
        if not os.path.exists(self.archive_path):
            raise FileNotFoundError(f"Neither local dataset directory ({self.dataset_dir}) nor archive zip ({self.archive_path}) found.")
        
        with zipfile.ZipFile(self.archive_path, 'r') as zf:
            with zf.open('movie_metadata/movie_meta_data.csv') as f:
                self._csv_content_cache = f.read().decode('utf-8', errors='ignore')
                return self._csv_content_cache

    def get_available_movies(self) -> List[Dict[str, str]]:
        movies = []
        content = self._get_metadata_csv_content()
        reader = csv.reader(io.StringIO(content))
        header = next(reader, None)
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

        content = self._get_metadata_csv_content()
        reader = csv.reader(io.StringIO(content))
        header = next(reader, None)
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

    def _load_raw_screenplay_json(self, padded_id: str, raw_id: str, clean_target_title: str) -> Optional[List[Any]]:
        # 1. Try local extracted directory first
        local_annot_dir = os.path.join(self.dataset_dir, "rule_based_annotations")
        if os.path.exists(local_annot_dir):
            files = os.listdir(local_annot_dir)
            matched_file = None
            for f in files:
                if f"_{padded_id}.json" in f or f"_{raw_id}.json" in f:
                    matched_file = f
                    break
            if not matched_file and clean_target_title:
                for f in files:
                    clean_file_title = re.sub(r'[^a-zA-Z0-9]', '', f.rsplit('_', 1)[0]).lower()
                    if clean_target_title in clean_file_title or clean_file_title in clean_target_title:
                        matched_file = f
                        break
            if matched_file:
                with open(os.path.join(local_annot_dir, matched_file), "r", encoding="utf-8", errors="ignore") as f:
                    return json.load(f)

        # 2. Fallback to archive zip
        if os.path.exists(self.archive_path):
            with zipfile.ZipFile(self.archive_path, 'r') as zf:
                namelist = zf.namelist()
                matched_file = None
                annot_files = [f for f in namelist if 'rule_based_annotations' in f and f.endswith('.json')]
                for f in annot_files:
                    basename = os.path.basename(f)
                    if f"_{padded_id}.json" in basename or f"_{raw_id}.json" in basename:
                        matched_file = f
                        break
                if not matched_file and clean_target_title:
                    for f in annot_files:
                        basename = os.path.basename(f)
                        clean_file_title = re.sub(r'[^a-zA-Z0-9]', '', basename.rsplit('_', 1)[0]).lower()
                        if clean_target_title in clean_file_title or clean_file_title in clean_target_title:
                            matched_file = f
                            break
                if matched_file:
                    with zf.open(matched_file) as f:
                        return json.loads(f.read().decode('utf-8', errors='ignore'))
        return None

    def load_screenplay_scenes(self, imdb_id_or_title: str) -> List[SceneSegment]:
        meta = self.load_movie_metadata(imdb_id_or_title)
        target_id = meta.imdb_id if meta else imdb_id_or_title
        raw_id = target_id.lower().replace('tt', '').strip()
        padded_id = raw_id.zfill(7)
        clean_target_title = re.sub(r'[^a-zA-Z0-9]', '', meta.title).lower() if meta else ""

        raw_json = self._load_raw_screenplay_json(padded_id, raw_id, clean_target_title)
        if not raw_json:
            return []

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

    def load_character_genders(self, imdb_id: str) -> Dict[str, str]:
        raw_id = imdb_id.lower().replace('tt', '').strip().zfill(7)
        try:
            import pickle
            # 1. Try local extracted directory first
            local_pickle = os.path.join(self.dataset_dir, "character_genders.pickle")
            if os.path.exists(local_pickle):
                with open(local_pickle, "rb") as f:
                    data = pickle.load(f)
                    matched = data.get(raw_id) or data.get(raw_id.lstrip('0'))
                    if matched:
                        return {c[0].title(): c[1] for c in matched if len(c) >= 2}

            # 2. Fallback to archive zip
            if os.path.exists(self.archive_path):
                with zipfile.ZipFile(self.archive_path, 'r') as zf:
                    if 'movie_characters/data/character_genders.pickle' in zf.namelist():
                        data = pickle.loads(zf.read('movie_characters/data/character_genders.pickle'))
                        matched = data.get(raw_id) or data.get(raw_id.lstrip('0'))
                        if matched:
                            return {c[0].title(): c[1] for c in matched if len(c) >= 2}
        except Exception:
            pass
        return {}
