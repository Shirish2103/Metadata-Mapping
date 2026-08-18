import pytest
from models.schema import TimeRange, SceneSegment, Dialogue, MovieMetadata
from extractors.character_presence_extractor import CharacterPresenceExtractor

def test_character_presence_extractor():
    # Setup test scenes with 4 scenes total
    scene1 = SceneSegment(
        scene_idx=1,
        location="OUTSIDE HOUSE",
        time_range=TimeRange(start_time="00:01:00", end_time="00:05:00"),
        dialogues=[
            Dialogue(speaker="HERO", text="We must leave now.", scene_idx=1),
            Dialogue(speaker="SIDEKICK", text="I agree with you.", scene_idx=1),
        ]
    )

    scene2 = SceneSegment(
        scene_idx=2,
        location="INSIDE CAR",
        time_range=TimeRange(start_time="00:06:00", end_time="00:10:00"),
        dialogues=[
            Dialogue(speaker="HERO", text="Faster!", scene_idx=2),
            Dialogue(speaker="VILLAIN", text="You cannot escape me.", scene_idx=2),
        ]
    )

    scene3 = SceneSegment(
        scene_idx=3,
        location="ALLEYWAY",
        time_range=TimeRange(start_time="00:11:00", end_time="00:15:00"),
        dialogues=[
            Dialogue(speaker="HERO", text="It's over.", scene_idx=3),
        ]
    )

    scene4 = SceneSegment(
        scene_idx=4,
        location="POLICE STATION",
        time_range=TimeRange(start_time="00:16:00", end_time="00:20:00"),
        dialogues=[
            Dialogue(speaker="HERO", text="Safe at last.", scene_idx=4),
            Dialogue(speaker="OFFICER", text="Good job.", scene_idx=4),
        ]
    )

    scenes = [scene1, scene2, scene3, scene4]
    config = {}
    extractor = CharacterPresenceExtractor(config)

    report = extractor.extract(movie_info=None, scenes=scenes)

    assert report.total_movie_scenes == 4
    assert len(report.characters) == 4

    # HERO appears in all 4 scenes (1,2,3,4) -> st_pct = 100.0%
    hero = next(c for c in report.characters if c.character_name == "Hero")
    assert hero.first_scene_idx == 1
    assert hero.last_scene_idx == 4
    assert hero.first_timestamp == "00:01:00"
    assert hero.last_timestamp == "00:20:00"
    assert hero.scene_count == 4
    assert hero.screen_time_percentage == 100.0
    assert hero.role_type == "Lead"

    # VILLAIN appears in scene 2 -> st_pct = 25.0%
    villain = next(c for c in report.characters if c.character_name == "Villain")
    assert villain.first_scene_idx == 2
    assert villain.last_scene_idx == 2
    assert villain.scene_count == 1
    assert villain.screen_time_percentage == 25.0
    assert villain.role_type == "Lead"

    # OFFICER appears only in scene 4 -> st_pct = 25.0%
    officer = next(c for c in report.characters if c.character_name == "Officer")
    assert officer.first_scene_idx == 4
    assert officer.last_scene_idx == 4
    assert officer.dialogue_line_count == 1
