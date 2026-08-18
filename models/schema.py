from typing import List, Optional
from pydantic import BaseModel, Field

class TimeRange(BaseModel):
    start_time: str = Field(description="Formatted timestamp e.g. 00:01:15")
    end_time: str = Field(description="Formatted timestamp e.g. 00:03:40")
    is_estimated: bool = Field(default=False, description="True if estimated via pacing rules")

class MovieMetadata(BaseModel):
    imdb_id: str
    title: str
    year: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    directors: List[str] = Field(default_factory=list)
    writers: List[str] = Field(default_factory=list)
    cast: List[str] = Field(default_factory=list)
    plot: Optional[str] = None

class Dialogue(BaseModel):
    speaker: str
    text: str
    scene_idx: int
    time_range: Optional[TimeRange] = None

class SceneSegment(BaseModel):
    scene_idx: int
    location: Optional[str] = None
    interior_exterior: Optional[str] = None  # INT. / EXT.
    time_of_day: Optional[str] = None        # DAY / NIGHT
    dialogues: List[Dialogue] = Field(default_factory=list)
    action_text: str = ""
    time_range: Optional[TimeRange] = None

class ExtractedTopics(BaseModel):
    main_topics: List[str] = Field(default_factory=list)
    subjects: List[str] = Field(default_factory=list)
    frequently_mentioned_terms: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

class ExtractedEntities(BaseModel):
    people: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)
    other_entities: List[str] = Field(default_factory=list)

class ExtractedSentiment(BaseModel):
    sentiment: str = Field(description="Positive, Negative, Neutral, or Mixed")
    emotions: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class ExtractedCategory(BaseModel):
    primary_category: str
    secondary_categories: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: Optional[str] = None

class DialogueEntry(BaseModel):
    timestamp: str = Field(default="00:00:00")
    speaker: str
    text: str
    location: Optional[str] = None

class ScriptMetadataResult(BaseModel):
    imdb_id: str
    title: str
    movie_info: Optional[MovieMetadata] = None
    time_range: Optional[TimeRange] = None
    topics: ExtractedTopics
    entities: ExtractedEntities
    sentiment: ExtractedSentiment
    category: ExtractedCategory
    scene_breakdown_count: int = 0
    speaker_list: List[str] = Field(default_factory=list)
    dialogues_in_window: List[DialogueEntry] = Field(default_factory=list)
