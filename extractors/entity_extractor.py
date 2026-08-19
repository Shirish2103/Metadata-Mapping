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
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def _is_valid_entity_nlp(self, entity_str: str) -> bool:
        if not entity_str or len(entity_str) <= 2:
            return False
        if entity_str in self._entity_cache:
            return self._entity_cache[entity_str]
        cand = re.sub(r'^[-*\s()"\':;.]+|[-*\s()"\':;.]+$', '', entity_str).strip()
        if not cand or len(cand) <= 2:
            self._entity_cache[entity_str] = False
            return False
        if self.nlp:
            doc = self.nlp(cand)
            for t in doc:
                if t.pos_ in {"NUM", "VERB", "PUNCT", "AUX", "DET", "ADP", "SCONJ", "CCONJ", "SYM"} or t.like_num or t.is_punct:
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

        active_text_sample = []
        for sc in scenes:
            if sc.location:
                loc_clean = re.sub(r'^(INT\.|EXT\.|INT/EXT\.|EXT/INT\.)\s*', '', sc.location, flags=re.IGNORECASE).strip()
                loc_clean = re.sub(r'^[-*\s()"\':;.]+|[-*\s()"\':;.]+$', '', loc_clean).strip()
                if self._is_valid_entity_nlp(loc_clean):
                    locations.add(loc_clean.title())

            if sc.action_text:
                active_text_sample.append(sc.action_text)

            for d in sc.dialogues:
                if d.text:
                    active_text_sample.append(d.text)
                if d.speaker and d.speaker.upper() != "UNKNOWN":
                    if self._is_valid_entity_nlp(d.speaker):
                        people.add(d.speaker.title())

        combined_sample = " ".join(active_text_sample[:10000])

        if movie_info and combined_sample:
            combined_sample_lower = combined_sample.lower()
            for person in (movie_info.cast + movie_info.directors + movie_info.writers):
                clean_p = re.sub(r'^[-*\s()]+|[-*\s()]+$', '', person).strip()
                if clean_p and len(clean_p) > 1 and self._is_valid_entity_nlp(clean_p):
                    first_name = clean_p.split()[0].lower()
                    last_name = clean_p.split()[-1].lower() if len(clean_p.split()) > 1 else ""
                    if len(first_name) > 2 and (first_name in combined_sample_lower or (last_name and len(last_name) > 2 and last_name in combined_sample_lower)):
                        people.add(clean_p.title())

        if self.nlp and combined_sample:
            doc = self.nlp(combined_sample[:15000])
            for ent in doc.ents:
                clean_ent = re.sub(r'^[-*\s()"\':;.]+|[-*\s()"\':;.]+$', '', ent.text).strip().title()
                if not self._is_valid_entity_nlp(clean_ent):
                    continue

                label = ent.label_
                if label == "PERSON":
                    if len(clean_ent.split()) <= 3:
                        people.add(clean_ent)
                elif label in {"GPE", "LOC", "FAC"}:
                    locations.add(clean_ent)
                elif label in {"ORG"}:
                    organizations.add(clean_ent)
                elif label in {"PRODUCT"}:
                    products.add(clean_ent)
                elif label in {"EVENT", "WORK_OF_ART", "LAW", "NORP"}:
                    other_entities.add(clean_ent)

        prompt = (
            f"Perform Named Entity Recognition (NER) on this transcript sample.\n"
            f"Title: {movie_info.title if movie_info else 'Unknown'}\n"
            f"Sample Text: {combined_sample[:1500]}\n"
            f"Identify real world entities and return JSON format:\n"
            f"{{\"people\": [...], \"organizations\": [...], \"locations\": [...], \"products\": [...], \"other_entities\": [...]}}"
        )
        llm_response = self.call_llm_with_timeout(prompt, timeout=5.0)
        if llm_response:
            try:
                import json
                json_str = re.sub(r'```json\s*|\s*```', '', llm_response).strip()
                parsed = json.loads(json_str)
                if "people" in parsed and isinstance(parsed["people"], list):
                    people.update([re.sub(r'^[-*\s()]+|[-*\s()]+$', '', str(x)).strip().title() for x in parsed["people"] if self._is_valid_entity_nlp(str(x))])
                if "organizations" in parsed and isinstance(parsed["organizations"], list):
                    organizations.update([str(x).strip().title() for x in parsed["organizations"] if self._is_valid_entity_nlp(str(x))])
                if "locations" in parsed and isinstance(parsed["locations"], list):
                    locations.update([str(x).strip().title() for x in parsed["locations"] if self._is_valid_entity_nlp(str(x))])
                if "products" in parsed and isinstance(parsed["products"], list):
                    products.update([str(x).strip().title() for x in parsed["products"] if self._is_valid_entity_nlp(str(x))])
                if "other_entities" in parsed and isinstance(parsed["other_entities"], list):
                    other_entities.update([str(x).strip().title() for x in parsed["other_entities"] if self._is_valid_entity_nlp(str(x))])
            except Exception:
                pass

        people_upper = {p.upper() for p in people}
        clean_locs = {loc for loc in locations if loc.upper() not in people_upper and len(loc) > 2}
        clean_orgs = {org for org in organizations if org.upper() not in people_upper and len(org) > 2}
        clean_others = {oth for oth in other_entities if oth.upper() not in people_upper and len(oth) > 2}

        return ExtractedEntities(
            people=sorted([p for p in people if self._is_valid_entity_nlp(p)])[:15],
            organizations=sorted([o for o in clean_orgs if self._is_valid_entity_nlp(o)])[:10],
            locations=sorted([l for l in clean_locs if self._is_valid_entity_nlp(l)])[:15],
            products=sorted([pr for pr in products if self._is_valid_entity_nlp(pr)])[:10],
            other_entities=sorted([ot for ot in clean_others if self._is_valid_entity_nlp(ot)])[:10]
        )
