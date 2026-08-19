import re
import os
from typing import List, Dict, Any, Optional, Set
from models.schema import MovieMetadata, SceneSegment, ExtractedSentiment
from extractors.base_extractor import BaseExtractor


class SentimentExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vader = None
        try:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except Exception as e:
            self.vader = None

    def extract(self, movie_info: Optional[MovieMetadata], scenes: List[SceneSegment]) -> ExtractedSentiment:
        full_text_list = []
        for sc in scenes:
            if sc.action_text:
                full_text_list.append(sc.action_text)
            for d in sc.dialogues:
                if d.text:
                    full_text_list.append(d.text)

        if not full_text_list and movie_info and movie_info.plot:
            full_text_list.append(movie_info.plot)

        if not full_text_list:
            return ExtractedSentiment(sentiment="Neutral", emotions=["Neutral"], confidence=0.50)

        # 1. Primary Extraction using NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner)
        if self.vader:
            chunks = full_text_list[:120]  # Process representative dialogue chunks
            compound_scores = []
            pos_scores = []
            neg_scores = []
            neu_scores = []

            for chunk in chunks:
                if chunk.strip():
                    vs = self.vader.polarity_scores(chunk)
                    compound_scores.append(vs['compound'])
                    pos_scores.append(vs['pos'])
                    neg_scores.append(vs['neg'])
                    neu_scores.append(vs['neu'])

            if compound_scores:
                avg_compound = sum(compound_scores) / len(compound_scores)
                avg_pos = sum(pos_scores) / len(pos_scores)
                avg_neg = sum(neg_scores) / len(neg_scores)
            else:
                avg_compound, avg_pos, avg_neg = 0.0, 0.0, 0.0

            # Determine Sentiment Label based on VADER Compound Score & Polarity Balance
            if avg_compound >= 0.05:
                if avg_neg > 0.12:
                    sentiment_label = "Mixed"
                else:
                    sentiment_label = "Positive"
            elif avg_compound <= -0.05:
                if avg_pos > 0.12:
                    sentiment_label = "Mixed"
                else:
                    sentiment_label = "Negative"
            else:
                if avg_pos > 0.08 and avg_neg > 0.08:
                    sentiment_label = "Mixed"
                else:
                    sentiment_label = "Neutral"

            # Derive Confidence Score from compound score magnitude
            confidence = min(0.98, max(0.65, round(abs(avg_compound) + 0.60, 2)))

            # Map Polarity Signals to Emotion Tags
            emotions = []
            if avg_pos > 0.20 or avg_compound >= 0.35:
                emotions.extend(["Joy", "Excitement", "Optimism"])
            elif avg_pos > 0.10 or avg_compound >= 0.05:
                emotions.extend(["Hopeful", "Satisfaction"])

            if avg_neg > 0.20 or avg_compound <= -0.35:
                emotions.extend(["Anger", "Fear", "Tension", "Grief"])
            elif avg_neg > 0.10 or avg_compound <= -0.05:
                emotions.extend(["Sadness", "Anxiety", "Melancholy"])

            if sentiment_label == "Mixed":
                emotions.extend(["Dramatic Tension", "Conflict"])

            if not emotions:
                emotions = ["Calm", "Neutral"]

            return ExtractedSentiment(
                sentiment=sentiment_label,
                emotions=sorted(list(set(emotions))),
                confidence=confidence
            )

        # Fallback to neutral if VADER is unavailable
        return ExtractedSentiment(
            sentiment="Neutral",
            emotions=["Neutral"],
            confidence=0.50
        )
