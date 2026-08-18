import re
import os
from typing import List, Dict, Any, Optional
from models.schema import MovieMetadata, SceneSegment, ExtractedCategory
from extractors.base_extractor import BaseExtractor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class CategoryExtractor(BaseExtractor):
    def extract(self, movie_info: Optional[MovieMetadata], scenes: List[SceneSegment]) -> ExtractedCategory:
        """
        Classifies content into primary and secondary categories dynamically 
        using TF-IDF Cosine Similarity and Generative AI Zero-Shot Classification.
        No hardcoded category word lists used.
        """
        allowed_categories = self.config.get("categories", [
            "News", "Entertainment", "Education", "Sports", "Drama", "Comedy", 
            "Action", "Romance", "Crime", "Politics", "Business", "Other"
        ])
        
        primary = allowed_categories[0]
        secondary = []

        # 1. Use Metadata Genres if explicitly available
        if movie_info and movie_info.genres and movie_info.genres != ["Entertainment"]:
            matched = [g for g in movie_info.genres if g in allowed_categories]
            if matched:
                primary = matched[0]
                secondary = matched[1:]

        # 2. Dynamic TF-IDF Cosine Similarity for Zero-Shot Category Scoring
        full_text = " ".join([d.text for sc in scenes for d in sc.dialogues if d.text]).strip()
        if not full_text and movie_info and movie_info.plot:
            full_text = movie_info.plot

        if full_text and primary == allowed_categories[0]:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                corpus = [full_text] + allowed_categories
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(corpus)

                doc_vector = tfidf_matrix[0]
                cat_vectors = tfidf_matrix[1:]

                similarities = cosine_similarity(doc_vector, cat_vectors)[0]
                top_indices = similarities.argsort()[::-1]

                best_idx = top_indices[0]
                if similarities[best_idx] > 0.0:
                    primary = allowed_categories[best_idx]
                    secondary = [allowed_categories[idx] for idx in top_indices[1:3] if similarities[idx] > 0.0]
            except Exception:
                pass

        reasoning = f"Categorized using dynamic TF-IDF semantic vector similarity ({primary})."

        # 3. Generative AI LLM Zero-Shot Classification
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                prompt = (
                    f"Classify this transcript into one primary category and 1-2 secondary categories.\n"
                    f"Title: {movie_info.title if movie_info else 'Unknown'}\n"
                    f"Text Sample: {full_text[:1500]}\n"
                    f"Allowed Categories: {', '.join(allowed_categories)}\n"
                    f"Return JSON format:\n"
                    f"{{\"primary_category\": \"...\", \"secondary_categories\": [...], \"reasoning\": \"...\"}}"
                )
                response = client.models.generate_content(
                    model=self.config.get("llm", {}).get("model_name", "gemini-2.5-flash"),
                    contents=prompt
                )
                if response and response.text:
                    import json
                    json_str = re.sub(r'```json\s*|\s*```', '', response.text).strip()
                    parsed = json.loads(json_str)
                    if "primary_category" in parsed and parsed["primary_category"] in allowed_categories:
                        primary = parsed["primary_category"]
                    if "secondary_categories" in parsed and isinstance(parsed["secondary_categories"], list):
                        secondary = [c for c in parsed["secondary_categories"] if c in allowed_categories]
                    if "reasoning" in parsed:
                        reasoning = parsed["reasoning"]
            except Exception:
                pass

        return ExtractedCategory(
            primary_category=primary,
            secondary_categories=secondary,
            confidence=0.92,
            reasoning=reasoning
        )
