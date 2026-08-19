import re
import os
from typing import List, Dict, Any, Optional, Set
from models.schema import MovieMetadata, SceneSegment, ExtractedEntities
from extractors.base_extractor import BaseExtractor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class EntityExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.nlp = None
        self._entity_cache: Dict[str, bool] = {}
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
        except Exception:
            self.nlp = None

    def _is_valid_entity_nlp(self, entity_str: str) -> bool:
        if not entity_str or len(entity_str) <= 2:
            return False
        if entity_str in self._entity_cache:
            return self._entity_cache[entity_str]

        cand = re.sub(r'^[-*\s()"\':;.]+|[-*\s()"\':;.]+$', '', entity_str).strip()
        if not cand or len(cand) <= 2 or len(cand.split()) > 4:
            self._entity_cache[entity_str] = False
            return False

        if self.nlp:
            doc = self.nlp(cand)
            # Pure POS & Entity Validation
            for t in doc:
                if t.pos_ in {"VERB", "AUX", "NUM", "PUNCT", "ADP", "SCONJ", "CCONJ", "SYM"} or t.like_num or t.is_punct:
                    self._entity_cache[entity_str] = False
                    return False
            for ent in doc.ents:
                if ent.label_ in {"DATE", "TIME", "CARDINAL", "QUANTITY", "PERCENT"}:
                    self._entity_cache[entity_str] = False
                    return False
        else:
            if re.search(r'\d+', cand) or any(c in cand for c in ['"', '...', '!', '?', ';']):
                self._entity_cache[entity_str] = False
                return False

        self._entity_cache[entity_str] = True
        return True

    def extract(self, movie_info: Optional[MovieMetadata], scenes: List[SceneSegment]) -> ExtractedEntities:
        people: Set[str] = set()
        locations: Set[str] = set()
        organizations: Set[str] = set()
        products: Set[str] = set()
        other_entities: Set[str] = set()

        full_text_list = []
        for sc in scenes:
            if sc.action_text:
                full_text_list.append(sc.action_text)
            for d in sc.dialogues:
                if d.text:
                    full_text_list.append(d.text)

        full_text = " ".join(full_text_list).strip()

        # Dynamic spaCy NER Extraction
        if full_text and self.nlp:
            text_sample = full_text[:5000]
            doc = self.nlp(text_sample)
            for ent in doc.ents:
                cleaned = ent.text.strip().title()
                if not self._is_valid_entity_nlp(cleaned):
                    continue

                if ent.label_ in {"PERSON"}:
                    people.add(cleaned)
                elif ent.label_ in {"GPE", "LOC", "FAC"}:
                    locations.add(cleaned)
                elif ent.label_ in {"ORG"}:
                    organizations.add(cleaned)
                elif ent.label_ in {"PRODUCT"}:
                    products.add(cleaned)
                elif ent.label_ in {"EVENT", "WORK_OF_ART", "NORP"}:
                    other_entities.add(cleaned)

        # Fallback dynamic regex NER when spaCy disabled
        if not people and not locations and full_text:
            capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', full_text[:3000])
            for w in capitalized_words:
                if self._is_valid_entity_nlp(w) and len(w) > 3:
                    people.add(w.title())

        # LLM zero-shot entity enrichment when API available
        prompt = (
            f"Extract Named Entities from this transcript text.\n"
            f"Title: {movie_info.title if movie_info else 'Unknown'}\n"
            f"Text Sample: {full_text[:1200]}\n"
            f"Return JSON format:\n"
            f"{{\"people\": [...], \"locations\": [...], \"organizations\": [...], \"products\": [...]}}"
        )
        llm_response = self.call_llm_with_timeout(prompt, timeout=5.0)
        if llm_response:
            try:
                import json
                json_str = re.sub(r'```json\s*|\s*```', '', llm_response).strip()
                parsed = json.loads(json_str)
                if "people" in parsed and isinstance(parsed["people"], list):
                    for p in parsed["people"]:
                        if self._is_valid_entity_nlp(str(p)): people.add(str(p).title())
                if "locations" in parsed and isinstance(parsed["locations"], list):
                    for l in parsed["locations"]:
                        if self._is_valid_entity_nlp(str(l)): locations.add(str(l).title())
                if "organizations" in parsed and isinstance(parsed["organizations"], list):
                    for o in parsed["organizations"]:
                        if self._is_valid_entity_nlp(str(o)): organizations.add(str(o).title())
                if "products" in parsed and isinstance(parsed["products"], list):
                    for pr in parsed["products"]:
                        if self._is_valid_entity_nlp(str(pr)): products.add(str(pr).title())
            except Exception:
                pass

        return ExtractedEntities(
            people=sorted(list(people))[:15],
            locations=sorted(list(locations))[:15],
            organizations=sorted(list(organizations))[:15],
            products=sorted(list(products))[:10],
            other_entities=sorted(list(other_entities))[:10]
        )
