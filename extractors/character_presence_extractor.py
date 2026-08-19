import re
from typing import List, Dict, Any, Optional
from extractors.base_extractor import BaseExtractor
from models.schema import MovieMetadata, SceneSegment, CharacterPresence, CharacterPresenceReport

NOISE_SPEAKER_WORDS = {
    "UNKNOWN", "NONE", "N/A", "CONTINUED", "TITLE", "SUPERIMPOSE", "MONTAGE",
    "SCENE", "CUT TO", "FADE IN", "FADE OUT", "DISSOLVE", "TIME DISSOLVE",
    "END MONTAGE", "END_MONTAGE", "MOMENTS LATER", "HALLWAY", "OFFSCREEN",
    "VOICEOVER", "CAMERA", "ANGLE", "CLOSE UP", "FLASHBACK", "EXT", "INT",
    "PROTAGONIST", "NARRATOR", "CHARACTER", "INTERVIEW", "AUDIENCE", "ACTOR",
    "ACTRESS", "ASSOCIATE", "COMPUTER VOICE"
}

def clean_and_validate_speaker(raw_speaker: str) -> Optional[str]:
    if not raw_speaker:
        return None
    
    cleaned = re.sub(r'\(.*?\)', '', raw_speaker)
    cleaned = re.sub(r"['’]s$", '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).title().strip()
    
    if not cleaned or len(cleaned) <= 1:
        return None
        
    upper_c = cleaned.upper()
    if upper_c in NOISE_SPEAKER_WORDS:
        return None
        
    for w in ["MONTAGE", "DISSOLVE", "SUPERIMPOSE", "CONTINUED"]:
        if w in upper_c:
            return None
            
    return cleaned

import os
import pickle

GENDER_MAP = {}
try:
    g_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset", "character_genders.pickle")
    if os.path.exists(g_path):
        with open(g_path, 'rb') as f:
            raw_genders = pickle.load(f)
            for imdb_id, char_list in raw_genders.items():
                clean_id = str(imdb_id).lstrip('0')
                for item in char_list:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        name_str = str(item[0])
                        gender_type = str(item[1]).lower()
                        g_label = "Female" if ("actress" in gender_type or "female" in gender_type) else ("Male" if ("actor" in gender_type or "male" in gender_type) else "Unspecified")
                        GENDER_MAP[(clean_id, name_str.lower())] = g_label
                        first_n = name_str.split()[0].lower() if name_str.split() else ""
                        if first_n:
                            GENDER_MAP[(clean_id, first_n)] = g_label
except Exception:
    pass

FEMALE_FIRST_NAMES = {"winona", "vickie", "cynthia", "patty", "janeane", "louise", "jennifer", "carol", "libby", "sarah", "mary", "elizabeth", "kate", "rachel", "emma", "anne", "claire", "julia", "laura"}
MALE_FIRST_NAMES = {"ethan", "steve", "ben", "grant", "pat", "dale", "phineas", "john", "dave", "lucien", "troy", "sammy", "michael", "luke", "roger", "david", "james", "robert", "michael", "william", "richard", "thomas", "charles", "christopher", "daniel", "matthew"}

def infer_character_gender(movie_imdb_id: Optional[str], character_name: str) -> str:
    clean_name = character_name.strip().lower()
    clean_id = str(movie_imdb_id).lstrip('0') if movie_imdb_id else ""
    
    if (clean_id, clean_name) in GENDER_MAP:
        return GENDER_MAP[(clean_id, clean_name)]
        
    first_n = clean_name.split()[0] if clean_name.split() else ""
    if (clean_id, first_n) in GENDER_MAP:
        return GENDER_MAP[(clean_id, first_n)]
        
    if first_n in FEMALE_FIRST_NAMES:
        return "Female"
    if first_n in MALE_FIRST_NAMES:
        return "Male"
        
    return "Unspecified"

class CharacterPresenceExtractor(BaseExtractor):
    def extract(
        self, 
        movie_info: Optional[MovieMetadata], 
        scenes: List[SceneSegment]
    ) -> CharacterPresenceReport:
        if not scenes:
            return CharacterPresenceReport(total_movie_scenes=0, characters=[])

        total_scenes = len(scenes)
        char_data: Dict[str, Dict[str, Any]] = {}
        movie_imdb_id = movie_info.imdb_id if movie_info else None

        for scene in scenes:
            scene_idx = scene.scene_idx
            start_ts = scene.time_range.start_time if scene.time_range else "00:00:00"
            end_ts = scene.time_range.end_time if scene.time_range else "00:00:00"

            for dialogue in scene.dialogues:
                speaker_name = clean_and_validate_speaker(dialogue.speaker)
                if not speaker_name:
                    continue

                if speaker_name not in char_data:
                    char_data[speaker_name] = {
                        "scenes_present": set(),
                        "dialogue_line_count": 0,
                        "first_scene_idx": scene_idx,
                        "last_scene_idx": scene_idx,
                        "first_timestamp": start_ts,
                        "last_timestamp": end_ts
                    }

                info = char_data[speaker_name]
                info["scenes_present"].add(scene_idx)
                info["dialogue_line_count"] += 1
                
                if scene_idx < info["first_scene_idx"]:
                    info["first_scene_idx"] = scene_idx
                    info["first_timestamp"] = start_ts
                if scene_idx > info["last_scene_idx"]:
                    info["last_scene_idx"] = scene_idx
                    info["last_timestamp"] = end_ts

        characters: List[CharacterPresence] = []
        for name, info in char_data.items():
            scene_count = len(info["scenes_present"])
            st_pct = round((scene_count / total_scenes) * 100.0, 2) if total_scenes > 0 else 0.0
            line_count = info["dialogue_line_count"]

            # Role classification
            if st_pct >= 15.0 or (st_pct >= 10.0 and line_count >= 25):
                role = "Lead"
            elif st_pct >= 5.0 or line_count >= 8:
                role = "Supporting"
            else:
                role = "Minor"

            gender_label = infer_character_gender(movie_imdb_id, name)

            characters.append(CharacterPresence(
                character_name=name,
                gender=gender_label,
                first_scene_idx=info["first_scene_idx"],
                last_scene_idx=info["last_scene_idx"],
                first_timestamp=info["first_timestamp"],
                last_timestamp=info["last_timestamp"],
                scene_count=scene_count,
                screen_time_percentage=st_pct,
                dialogue_line_count=line_count,
                role_type=role
            ))

        # Sort characters by screen time percentage descending, then line count
        characters.sort(key=lambda c: (c.screen_time_percentage, c.dialogue_line_count), reverse=True)

        return CharacterPresenceReport(
            total_movie_scenes=total_scenes,
            characters=characters
        )
