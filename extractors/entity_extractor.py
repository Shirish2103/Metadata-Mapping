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
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def extract(self, movie_info: Optional[MovieMetadata], scenes: List[SceneSegment]) -> ExtractedEntities:
        """
        Extracts named entities (People, Organizations, Locations, Products, Other) 
        using spaCy Statistical NER pipeline, POS Proper Noun tagging, and Generative AI.
        No hardcoded entity dictionaries used.
        """
        people: Set[str] = set()
        locations: Set[str] = set()
        organizations: Set[str] = set()
        products: Set[str] = set()
        other_entities: Set[str] = set()

        # 1. Structural extraction: Metadata cast, directors, writers & scene speakers
        if movie_info:
            for person in (movie_info.cast + movie_info.directors + movie_info.writers):
                clean_p = re.sub(r'^[-*\s()]+|[-*\s()]+$', '', person).strip()
                if clean_p and len(clean_p) > 1:
                    people.add(clean_p.title())

        for sc in scenes:
            if sc.location:
                loc_clean = re.sub(r'^(INT\.|EXT\.|INT/EXT\.)\s*', '', sc.location, flags=re.IGNORECASE).strip()
                loc_clean = re.sub(r'^[-*\s()]+|[-*\s()]+$', '', loc_clean).strip()
                if len(loc_clean) > 2 and loc_clean.upper() not in {"SCENE", "CONTINUED", "DARKNESS", "TRANSCRIPT"}:
                    locations.add(loc_clean.title())

            for d in sc.dialogues:
                if d.speaker:
                    spk = re.sub(r'^[-*\s()]+|[-*\s()]+$', '', d.speaker).strip()
                    if spk and len(spk) > 1 and spk.upper() not in {"UNKNOWN", "VOICE"}:
                        people.add(spk.title())

        # 2. spaCy Statistical Named Entity Recognition (NER)
        sample_texts = []
        for sc in scenes[:25]:
            if sc.action_text:
                sample_texts.append(sc.action_text)
            for d in sc.dialogues:
                sample_texts.append(d.text)

        if not sample_texts and movie_info and movie_info.plot:
            sample_texts.append(movie_info.plot)

        combined_sample = " ".join(sample_texts[:8000])

        if self.nlp and combined_sample:
            doc = self.nlp(combined_sample[:15000])
            for ent in doc.ents:
                clean_ent = re.sub(r'^[-*\s()]+|[-*\s()]+$', '', ent.text).strip().title()
                if len(clean_ent) <= 2 or ent.text.isupper() and len(ent.text.split()) > 3:
                    continue

                label = ent.label_
                if label == "PERSON":
                    people.add(clean_ent)
                elif label in {"GPE", "LOC", "FAC"}:
                    locations.add(clean_ent)
                elif label in {"ORG"}:
                    organizations.add(clean_ent)
                elif label in {"PRODUCT"}:
                    products.add(clean_ent)
                elif label in {"EVENT", "WORK_OF_ART", "LAW", "NORP"}:
                    other_entities.add(clean_ent)

        # 3. Generative AI LLM Zero-Shot NER (if API Key present)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                prompt = (
                    f"Perform Named Entity Recognition (NER) on this transcript sample.\n"
                    f"Title: {movie_info.title if movie_info else 'Unknown'}\n"
                    f"Sample Text: {combined_sample[:1500]}\n"
                    f"Identify and return JSON format:\n"
                    f"{{\"people\": [...], \"organizations\": [...], \"locations\": [...], \"products\": [...], \"other_entities\": [...]}}"
                )
                response = client.models.generate_content(
                    model=self.config.get("llm", {}).get("model_name", "gemini-2.5-flash"),
                    contents=prompt
                )
                if response and response.text:
                    import json
                    json_str = re.sub(r'```json\s*|\s*```', '', response.text).strip()
                    parsed = json.loads(json_str)
                    if "people" in parsed and isinstance(parsed["people"], list):
                        people.update([re.sub(r'^[-*\s()]+|[-*\s()]+$', '', str(x)).strip().title() for x in parsed["people"]])
                    if "organizations" in parsed and isinstance(parsed["organizations"], list):
                        organizations.update([str(x).strip() for x in parsed["organizations"]])
                    if "locations" in parsed and isinstance(parsed["locations"], list):
                        locations.update([str(x).strip().title() for x in parsed["locations"]])
                    if "products" in parsed and isinstance(parsed["products"], list):
                        products.update([str(x).strip().title() for x in parsed["products"]])
                    if "other_entities" in parsed and isinstance(parsed["other_entities"], list):
                        other_entities.update([str(x).strip().title() for x in parsed["other_entities"]])
            except Exception:
                pass  # Fallback to spaCy NER

        # Dynamic cross-category deduplication (Person names priority over Location/Org)
        people_upper = {p.upper() for p in people}
        clean_locs = {loc for loc in locations if loc.upper() not in people_upper and len(loc) > 2}
        clean_orgs = {org for org in organizations if org.upper() not in people_upper and len(org) > 2}

        return ExtractedEntities(
            people=sorted(list(people))[:15],
            organizations=sorted(list(clean_orgs))[:10],
            locations=sorted(list(clean_locs))[:15],
            products=sorted(list(products))[:10],
            other_entities=sorted(list(other_entities))[:10]
        )
