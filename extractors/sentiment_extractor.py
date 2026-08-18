import re
import os
from typing import List, Dict, Any, Optional, Set
from models.schema import MovieMetadata, SceneSegment, ExtractedSentiment
from extractors.base_extractor import BaseExtractor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class SentimentExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def extract(self, movie_info: Optional[MovieMetadata], scenes: List[SceneSegment]) -> ExtractedSentiment:
        full_text_list = []
        for sc in scenes:
            if sc.action_text:
                full_text_list.append(sc.action_text)
            for d in sc.dialogues:
                full_text_list.append(d.text)

        if not full_text_list and movie_info and movie_info.plot:
            full_text_list.append(movie_info.plot)

        combined_text = " ".join(full_text_list[:30])
        if not combined_text.strip():
            return ExtractedSentiment(sentiment="Neutral", emotions=[], confidence=0.5)

        pos_count = 0
        neg_count = 0
        emotions_found: Set[str] = set()

        if self.nlp:
            doc = self.nlp(combined_text[:5000])
            for token in doc:
                if token.pos_ in {"ADJ", "ADV", "VERB"}:
                    lemma = token.lemma_.lower()
                    if token.sentiment > 0:
                        pos_count += 1
                    elif token.sentiment < 0:
                        neg_count += 1

        total = pos_count + neg_count
        if total > 0:
            ratio = pos_count / total
            if ratio > 0.65:
                overall_sentiment = "Positive"
            elif ratio < 0.35:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Mixed"
        else:
            overall_sentiment = "Neutral"

        confidence = 0.85

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                prompt = (
                    f"Perform Sentiment and Multi-Label Emotion Analysis on this transcript text.\n"
                    f"Title: {movie_info.title if movie_info else 'Unknown'}\n"
                    f"Text Sample: {combined_text[:1500]}\n"
                    f"Identify:\n"
                    f"- overall sentiment: ('Positive', 'Negative', 'Neutral', or 'Mixed')\n"
                    f"- emotions: list of detected emotions (e.g. Happiness, Sadness, Anger, Fear, Surprise, Love, Excitement, Anxiety, Determination)\n"
                    f"- confidence score (float 0.0 to 1.0)\n"
                    f"Return JSON format:\n"
                    f"{{\"sentiment\": \"...\", \"emotions\": [...], \"confidence\": 0.88}}"
                )
                response = client.models.generate_content(
                    model=self.config.get("llm", {}).get("model_name", "gemini-3.6-flash"),
                    contents=prompt
                )
                if response and response.text:
                    import json
                    json_str = re.sub(r'```json\s*|\s*```', '', response.text).strip()
                    parsed = json.loads(json_str)
                    if "sentiment" in parsed:
                        overall_sentiment = parsed["sentiment"]
                    if "emotions" in parsed and isinstance(parsed["emotions"], list):
                        emotions_found = set(parsed["emotions"])
                    if "confidence" in parsed and isinstance(parsed["confidence"], (int, float)):
                        confidence = float(parsed["confidence"])
            except Exception:
                pass

        if not emotions_found:
            emotions_found = {"Happiness", "Excitement"} if overall_sentiment == "Positive" else {"Anger", "Fear"} if overall_sentiment == "Negative" else {"Neutral"}

        return ExtractedSentiment(
            sentiment=overall_sentiment,
            emotions=sorted(list(emotions_found)),
            confidence=confidence
        )
