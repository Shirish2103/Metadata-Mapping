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

    def _is_valid_speaker_nlp(self, speaker_candidate: str) -> Optional[str]:
        if not speaker_candidate:
            return None
        if speaker_candidate in self._speaker_cache:
            return self._speaker_cache[speaker_candidate]

        # Fast Regex Sanitization & Noise Purge
        cand = re.sub(r'\(.*?\)', '', str(speaker_candidate))
        cand = re.sub(r"['’]s$", '', cand, flags=re.IGNORECASE)
        cand = re.sub(r'[^a-zA-Z\s]', ' ', cand).strip()
        cand = re.sub(r'\s+', ' ', cand).title().strip()
        
        if not cand or len(cand) <= 1 or len(cand.split()) > 3:
            self._speaker_cache[speaker_candidate] = None
            return None

        upper_c = cand.upper()
        noise_terms = {
            "UNKNOWN", "NONE", "N/A", "CONTINUED", "TITLE", "SUPERIMPOSE", "MONTAGE",
            "SCENE", "CUT TO", "FADE IN", "FADE OUT", "DISSOLVE", "TIME DISSOLVE",
            "END MONTAGE", "END_MONTAGE", "MOMENTS LATER", "HALLWAY", "OFFSCREEN",
            "VOICEOVER", "CAMERA", "ANGLE", "CLOSE UP", "FLASHBACK", "EXT", "INT"
        }
        if upper_c in noise_terms:
            self._speaker_cache[speaker_candidate] = None
            return None

        for w in ["MONTAGE", "DISSOLVE", "SUPERIMPOSE", "CONTINUED"]:
            if w in upper_c:
                self._speaker_cache[speaker_candidate] = None
                return None

        self._speaker_cache[speaker_candidate] = cand
        return cand

    def _get_metadata_csv_content(self) -> str:
        if self._csv_content_cache is not None:
            return self._csv_content_cache

        # 1. Try local extracted directory first
        local_csv = os.path.join(self.dataset_dir, "movie_metadata", "movie_meta_data.csv")
        if os.path.exists(local_csv):
            with open(local_csv, "r", encoding="utf-8", errors="ignore") as f:
                self._csv_content_cache = f.read()
                return self._csv_content_cache

        # 2. Try candidate zip archive locations
        candidate_paths = [
            self.archive_path,
            os.path.join("data", "archive.zip"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "archive.zip"),
            r"d:\Downloads\archive.zip"
        ]

        found_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                found_path = p
                break

        if found_path:
            with zipfile.ZipFile(found_path, 'r') as zf:
                with zf.open('movie_metadata/movie_meta_data.csv') as f:
                    self._csv_content_cache = f.read().decode('utf-8', errors='ignore')
                    return self._csv_content_cache

        # Return fallback CSV header if no dataset zip present on cloud
        self._csv_content_cache = "movie_id,movie_title,year,score,votes,language,country,cover_url,director,cast,genres,plot"
        return self._csv_content_cache

    def get_available_movies(self) -> List[Dict[str, str]]:
        # 1. Try pre-indexed JSON file first
        json_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "movies_index.json"),
            os.path.join("data", "movies_index.json")
        ]
        for jp in json_paths:
            if os.path.exists(jp):
                try:
                    with open(jp, 'r', encoding='utf-8', errors='ignore') as f:
                        data = json.load(f)
                        if data:
                            return data
                except Exception:
                    pass

        # 2. Fallback to CSV scanning
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
        raw_target = imdb_id_or_title.strip()
        target = raw_target.lower()
        clean_target = re.sub(r'\s*\(\d{4}\)', '', target).strip()
        target_id = target.rsplit('_', 1)[-1] if '_' in target else target
        clean_target_alnum = re.sub(r'[^a-zA-Z0-9]', '', clean_target)

        # 1. Try CSV metadata scanning first
        content = self._get_metadata_csv_content()
        reader = csv.reader(io.StringIO(content))
        header = next(reader, None)
        rows = [r for r in reader if r and len(r) >= 2]

        matched_row = None
        # Pass 1: Exact IMDB ID or Exact Title match
        for row in rows:
            imdb_id = row[0].strip().lower()
            title = row[1].strip().lower()
            clean_title_alnum = re.sub(r'[^a-zA-Z0-9]', '', title)
            if imdb_id == target or imdb_id == target_id or title == target or title == clean_target or clean_title_alnum == clean_target_alnum:
                matched_row = row
                break

        if matched_row:
            genres = [g.strip() for g in matched_row[22].split(',')] if len(matched_row) > 22 and matched_row[22] else []
            directors = [d.strip() for d in matched_row[14].split(',')] if len(matched_row) > 14 and matched_row[14] else []
            writers = [w.strip() for w in matched_row[13].split(',')] if len(matched_row) > 13 and matched_row[13] else []
            cast = [c.strip() for c in matched_row[16].split(',')] if len(matched_row) > 16 and matched_row[16] else []
            plot = matched_row[19].strip() if len(matched_row) > 19 else ""

            return MovieMetadata(
                imdb_id=matched_row[0].strip(),
                title=matched_row[1].strip(),
                year=matched_row[3].strip() if len(matched_row) > 3 else None,
                genres=genres,
                directors=directors,
                writers=writers,
                cast=cast,
                plot=plot
            )

        # 2. Fallback to movies_index.json metadata
        available = self.get_available_movies()
        for m in available:
            m_title = m.get("title", "").strip()
            m_id = str(m.get("imdb_id", "")).strip()
            clean_m_title = re.sub(r'[^a-zA-Z0-9]', '', m_title.lower())
            if (
                m_title.lower() == target 
                or m_title.lower() == clean_target
                or clean_m_title == clean_target_alnum
                or m_id.lower() == target 
                or m_id.lower() == target_id 
            ):
                genres_list = [g.strip() for g in m.get("genres", "").split(',') if g.strip()]
                return MovieMetadata(
                    imdb_id=m_id,
                    title=m_title,
                    year=m.get("year", "2026"),
                    genres=genres_list if genres_list else ["Drama"],
                    directors=["Director"],
                    writers=["Writer"],
                    cast=["Cast"],
                    plot=f"Screenplay transcript analysis for {m_title}."
                )

        return None

    _ZIP_MAPPING_CACHE: Optional[Dict[str, str]] = None
    _ZIP_REF: Optional[zipfile.ZipFile] = None

    def _get_zip_mapping(self) -> Dict[str, str]:
        if DatasetLoader._ZIP_MAPPING_CACHE is None and os.path.exists(self.archive_path):
            try:
                DatasetLoader._ZIP_REF = zipfile.ZipFile(self.archive_path, 'r')
                namelist = DatasetLoader._ZIP_REF.namelist()
                mapping = {}
                annot_files = [f for f in namelist if 'rule_based_annotations' in f and f.endswith('.json')]
                for f in annot_files:
                    basename = os.path.basename(f)
                    file_id = basename.rsplit('_', 1)[-1].replace('.json', '').lower()
                    raw_id = file_id.replace('tt', '').strip()
                    padded_id = raw_id.zfill(7)
                    
                    title_part = basename.rsplit('_', 1)[0].lower()
                    clean_t = re.sub(r'[^a-zA-Z0-9]', '', title_part)

                    mapping[f"id_{padded_id}"] = f
                    mapping[f"id_{raw_id}"] = f
                    if clean_t:
                        mapping[f"title_{clean_t}"] = f
                DatasetLoader._ZIP_MAPPING_CACHE = mapping
            except Exception:
                DatasetLoader._ZIP_MAPPING_CACHE = {}
        return DatasetLoader._ZIP_MAPPING_CACHE or {}

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
                    if clean_file_title == clean_target_title:
                        matched_file = f
                        break
            if matched_file:
                with open(os.path.join(local_annot_dir, matched_file), "r", encoding="utf-8", errors="ignore") as f:
                    return json.load(f)

        # 2. Fast O(1) Zip Lookup from in-memory cache
        mapping = self._get_zip_mapping()
        matched_file = (
            mapping.get(f"id_{padded_id}") or 
            mapping.get(f"id_{raw_id}") or 
            (mapping.get(f"title_{clean_target_title}") if clean_target_title else None)
        )

        if matched_file and DatasetLoader._ZIP_REF:
            try:
                with DatasetLoader._ZIP_REF.open(matched_file) as f:
                    return json.loads(f.read().decode('utf-8', errors='ignore'))
            except Exception:
                pass
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
