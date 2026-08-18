from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from models.schema import MovieMetadata, SceneSegment

class BaseExtractor(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def extract(
        self, 
        movie_info: Optional[MovieMetadata], 
        scenes: List[SceneSegment]
    ) -> Any:
        """Abstract method to extract metadata from a movie and its scenes."""
        pass
