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
            self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
        except Exception:
            self.nlp = None

    def extract(self, movie_info: Optional[MovieMetadata], scenes: List[SceneSegment]) -> ExtractedTopics:
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

        noun_lemmas = []
        doc_chunks = []

        if self.nlp:
            doc = self.nlp(combined_text[:5000])
            
            for token in doc:
                if (
                    token.pos_ in {"NOUN", "PROPN"}
                    and not token.is_stop
                    and not token.is_punct
                    and not token.like_num
                    and len(token.lemma_) > 2
                    and token.ent_type_ not in {"PERSON", "DATE", "TIME", "CARDINAL", "QUANTITY", "PERCENT"}
                    and token.dep_ in {"nsubj", "dobj", "pobj", "ROOT", "attr", "compound"}
                ):
                    noun_lemmas.append(token.lemma_.lower())

        freq_terms = []
        keywords = []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
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
            counts = Counter(noun_lemmas)
            ranked = [word for word, _ in counts.most_common(20)]
            keywords = ranked[:15]
            freq_terms = ranked[:10]

        main_topics = []
        if movie_info and movie_info.genres:
            for g in movie_info.genres:
                if g not in main_topics and g.lower() not in {"entertainment", "other"}:
                    main_topics.append(g)

        subjects = [kw.title() for kw in keywords[2:8]] if len(keywords) >= 8 else [kw.title() for kw in keywords[:5]]

        prompt = (
            f"Perform high-level thematic topic extraction on this movie transcript.\n"
            f"Title: {movie_info.title if movie_info else 'Unknown'}\n"
            f"Plot/Sample Text: {combined_text[:2000]}\n"
            f"Top NLP Keywords: {', '.join(keywords[:10])}\n\n"
            f"Return ONLY valid JSON with abstract thematic topics and subjects (exclude character names, numbers, script directions):\n"
            f"{{\"main_topics\": [3-5 abstract themes/genres], \"subjects\": [4-6 specific sub-topics], \"keywords\": [8-12 thematic keywords]}}"
        )
        llm_response = self.call_llm_with_timeout(prompt, timeout=5.0)
        if llm_response:
            try:
                import json
                json_str = re.sub(r'```json\s*|\s*```', '', llm_response).strip()
                parsed = json.loads(json_str)
                if "main_topics" in parsed and isinstance(parsed["main_topics"], list) and parsed["main_topics"]:
                    main_topics = [t.title() for t in parsed["main_topics"] if isinstance(t, str)]
                if "subjects" in parsed and isinstance(parsed["subjects"], list) and parsed["subjects"]:
                    subjects = [s.title() for s in parsed["subjects"] if isinstance(s, str)]
                if "keywords" in parsed and isinstance(parsed["keywords"], list) and parsed["keywords"]:
                    keywords = [k.lower() for k in parsed["keywords"] if isinstance(k, str)]
            except Exception:
                pass

        return ExtractedTopics(
            main_topics=main_topics[:6],
            subjects=subjects[:8],
            frequently_mentioned_terms=freq_terms[:10],
            keywords=keywords[:15]
        )
