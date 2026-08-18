import re
import os
from typing import List, Dict, Any, Optional, Set
from models.schema import MovieMetadata, SceneSegment, ExtractedTopics
from extractors.base_extractor import BaseExtractor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class TopicExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def extract(self, movie_info: Optional[MovieMetadata], scenes: List[SceneSegment]) -> ExtractedTopics:
        """
        Extracts topics, subjects, and keywords using TF-IDF Vectorization, 
        spaCy POS tagging (Nouns/Adjectives), Lemmatization, and LLM Generative AI.
        No hardcoded word lists used.
        """
        full_text_list = []
        for sc in scenes:
            if sc.action_text:
                full_text_list.append(sc.action_text)
            for d in sc.dialogues:
                full_text_list.append(d.text)

        if not full_text_list and movie_info and movie_info.plot:
            full_text_list.append(movie_info.plot)

        combined_text = " ".join(full_text_list)
        if not combined_text.strip():
            return ExtractedTopics(main_topics=[], subjects=[], frequently_mentioned_terms=[], keywords=[])

        # 1. NLP Processing using spaCy (POS Tagging & Lemmatization)
        lemmatized_tokens = []
        noun_adj_tokens = []
        if self.nlp:
            doc = self.nlp(combined_text[:30000])
            for token in doc:
                if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num and len(token.lemma_) > 2:
                    lemma = token.lemma_.lower()
                    lemmatized_tokens.append(lemma)
                    if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
                        noun_adj_tokens.append(lemma)
        else:
            words = re.findall(r'\b[A-Za-z]{3,}\b', combined_text.lower())
            lemmatized_tokens = words
            noun_adj_tokens = words

        # 2. Dynamic TF-IDF Calculation for Keywords & Phrase Extraction
        freq_terms = []
        keywords = []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Split scene texts into doc chunks for TF-IDF calculation
            doc_chunks = [sc.action_text + " " + " ".join([d.text for d in sc.dialogues]) for sc in scenes if sc.action_text or sc.dialogues]
            if not doc_chunks:
                doc_chunks = [combined_text]

            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                token_pattern=r'\b[a-zA-Z]{3,}\b',
                max_features=50
            )
            tfidf_matrix = vectorizer.fit_transform(doc_chunks)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            ranked_indices = scores.argsort()[::-1]

            ranked_terms = [feature_names[i] for i in ranked_indices]
            keywords = [term for term in ranked_terms if len(term) > 2][:15]
            freq_terms = keywords[:10]
        except Exception:
            from collections import Counter
            counts = Counter(noun_adj_tokens)
            ranked = [word for word, _ in counts.most_common(20)]
            keywords = ranked[:15]
            freq_terms = ranked[:10]

        # 3. Dynamic Topic & Subject Grouping
        main_topics = []
        if movie_info and movie_info.genres:
            for g in movie_info.genres:
                if g not in main_topics and g.lower() not in {"entertainment", "other"}:
                    main_topics.append(g)

        for kw in keywords:
            kw_title = kw.title()
            if kw_title not in main_topics:
                main_topics.append(kw_title)
            if len(main_topics) >= 5:
                break

        subjects = [kw.title() for kw in keywords[3:10]] if len(keywords) >= 8 else [kw.title() for kw in keywords[:5]]

        # 4. Generative AI LLM Zero-Shot Enhancement (if API Key present)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                prompt = (
                    f"Perform NLP topic extraction on this transcript text.\n"
                    f"Title: {movie_info.title if movie_info else 'Unknown'}\n"
                    f"Text Sample: {combined_text[:2000]}\n"
                    f"Top TF-IDF Terms: {', '.join(keywords[:10])}\n"
                    f"Extract high-level thematic metadata in JSON format:\n"
                    f"{{\"main_topics\": [3-5 main themes], \"subjects\": [4-6 specific subjects], \"keywords\": [8-12 keywords]}}"
                )
                response = client.models.generate_content(
                    model=self.config.get("llm", {}).get("model_name", "gemini-2.5-flash"),
                    contents=prompt
                )
                if response and response.text:
                    import json
                    json_str = re.sub(r'```json\s*|\s*```', '', response.text).strip()
                    parsed = json.loads(json_str)
                    if "main_topics" in parsed and isinstance(parsed["main_topics"], list):
                        main_topics = parsed["main_topics"]
                    if "subjects" in parsed and isinstance(parsed["subjects"], list):
                        subjects = parsed["subjects"]
                    if "keywords" in parsed and isinstance(parsed["keywords"], list):
                        keywords = parsed["keywords"]
            except Exception:
                pass  # Fallback to pure NLP output

        return ExtractedTopics(
            main_topics=main_topics[:6],
            subjects=subjects[:8],
            frequently_mentioned_terms=freq_terms[:10],
            keywords=keywords[:15]
        )
