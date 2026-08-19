import os
import concurrent.futures
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

    def call_llm_with_timeout(self, prompt: str, timeout: float = 5.0) -> Optional[str]:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None

        model_name = self.config.get("llm", {}).get("model_name", "gemini-2.5-flash")

        def _do_call():
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    return response.text
            except Exception:
                pass
            return None

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_call)
                return future.result(timeout=timeout)
        except Exception:
            return None
