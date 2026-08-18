from typing import List, Dict, Any, Optional
from extractors.base_extractor import BaseExtractor
from models.schema import MovieMetadata, SceneSegment, CharacterPresence, CharacterPresenceReport

class CharacterPresenceExtractor(BaseExtractor):
    def extract(
        self, 
        movie_info: Optional[MovieMetadata], 
        scenes: List[SceneSegment]
    ) -> CharacterPresenceReport:
        if not scenes:
            return CharacterPresenceReport(total_movie_scenes=0, characters=[])

        total_scenes = len(scenes)
        
        # Track character metrics
        # char_name -> dict of info
        char_data: Dict[str, Dict[str, Any]] = {}

        for scene in scenes:
            scene_idx = scene.scene_idx
            start_ts = scene.time_range.start_time if scene.time_range else "00:00:00"
            end_ts = scene.time_range.end_time if scene.time_range else "00:00:00"

            for dialogue in scene.dialogues:
                speaker = dialogue.speaker.strip() if dialogue.speaker else None
                if not speaker or speaker.lower() in {"unknown", "none"}:
                    continue
                
                # Normalize speaker name to Title Case
                speaker_name = speaker.title()

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

            characters.append(CharacterPresence(
                character_name=name,
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
