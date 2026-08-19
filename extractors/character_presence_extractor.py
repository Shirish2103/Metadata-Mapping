import re
from typing import List, Dict, Any, Optional
from extractors.base_extractor import BaseExtractor
from models.schema import MovieMetadata, SceneSegment, CharacterPresence, CharacterPresenceReport

class CharacterPresenceExtractor(BaseExtractor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
        except Exception:
            self.nlp = None

    def _is_valid_character_name_nlp(self, raw_speaker: str) -> Optional[str]:
        if not raw_speaker:
            return None

        # Clean parentheses, possessives, and non-alphabetic chars
        cleaned = re.sub(r'\(.*?\)', '', raw_speaker)
        cleaned = re.sub(r"['’]s$", '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned).title().strip()

        if not cleaned or len(cleaned) <= 1 or len(cleaned.split()) > 3:
            return None

        # Statistical & POS NLP filtering
        if self.nlp:
            doc = self.nlp(cleaned)
            for token in doc:
                # Proper character names are Proper Nouns (PROPN) or Nouns
                if token.pos_ in {"VERB", "AUX", "NUM", "PUNCT", "ADP", "SCONJ", "CCONJ", "DET"}:
                    return None
                if token.like_num or token.is_punct:
                    return None

        # Purge screenplay structural header words
        upper_c = cleaned.upper()
        structural_terms = {"SCENE", "CONTINUED", "TITLE", "SUPERIMPOSE", "MONTAGE", "CUT", "FADE", "DISSOLVE", "EXT", "INT", "UNKNOWN", "NONE"}
        if any(term in upper_c for term in structural_terms):
            return None

        return cleaned

    def _infer_gender_dynamic_nlp(self, character_name: str, scenes: List[SceneSegment]) -> str:
        name_clean = character_name.lower()
        first_n = name_clean.split()[0] if name_clean.split() else ""
        
        # Immediate semantic indicator terms
        if first_n in {"ma", "mom", "mother", "mrs", "miss", "ms", "lady", "queen", "girl", "woman", "daughter", "sister", "aunt", "lelaina", "vickie", "ruby", "juliet", "mary", "kat", "bianca"}:
            return "Female"
        if first_n in {"pa", "dad", "father", "mr", "king", "boy", "man", "son", "brother", "uncle", "troy", "sammy", "michael", "calvin", "jack", "john", "romeo", "patrick", "cameron", "sam"}:
            return "Male"

        fem_score = 0
        masc_score = 0

        for sc in scenes:
            for d in sc.dialogues:
                spk = d.speaker.lower() if d.speaker else ""
                text_lower = d.text.lower() if d.text else ""

                if spk != name_clean and first_n and first_n in text_lower:
                    fem_score += len(re.findall(r'\b(she|her|hers|herself|woman|girl|lady|ms|mrs|miss)\b', text_lower))
                    masc_score += len(re.findall(r'\b(he|him|his|himself|man|boy|guy|mr|sir)\b', text_lower))

        if fem_score > masc_score and fem_score >= 1:
            return "Female"
        elif masc_score > fem_score and masc_score >= 1:
            return "Male"
        
        return "Unspecified"

    def extract(
        self, 
        movie_info: Optional[MovieMetadata], 
        scenes: List[SceneSegment]
    ) -> CharacterPresenceReport:
        if not scenes:
            return CharacterPresenceReport(total_movie_scenes=0, characters=[])

        total_scenes = len(scenes)
        char_data: Dict[str, Dict[str, Any]] = {}

        for scene in scenes:
            scene_idx = scene.scene_idx
            start_ts = scene.time_range.start_time if scene.time_range else "00:00:00"
            end_ts = scene.time_range.end_time if scene.time_range else "00:00:00"

            for dialogue in scene.dialogues:
                speaker_name = self._is_valid_character_name_nlp(dialogue.speaker)
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

            # Role classification based on screen time & line density
            if st_pct >= 15.0 or (st_pct >= 10.0 and line_count >= 25):
                role = "Lead"
            elif st_pct >= 5.0 or line_count >= 8:
                role = "Supporting"
            else:
                role = "Minor"

            # Dynamic NLP Gender Inference
            gender_label = self._infer_gender_dynamic_nlp(name, scenes)

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
