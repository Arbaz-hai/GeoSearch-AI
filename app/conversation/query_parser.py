from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any



_CHANGE_KEYWORDS = {
    "construction": {"construction", "build", "building", "built", "new building", "new road"},
    "clearance": {"clearance", "cleared", "deforested", "deforestation", "logging"},
    "water": {"flood", "flooding", "inundation", "water expansion", "water body", "lake", "river"},
    "road": {"road", "highway", "motorway", "road expansion", "road widening"},
    "urban_expansion": {"urban", "urbanization", "urban expansion", "city growth", "sprawl"},
    "vegetation": {"vegetation", "ndvi", "forest", "trees", "crop", "farmland"},
    "appearance": {"appeared", "new", "newly", "emerged"},
    "disappearance": {"disappeared", "removed", "demolished", "gone"},
}

_OBJECT_TERMS = {
    "river", "waterway", "lake", "reservoir", "sea", "ocean", "coast",
    "road", "highway", "motorway", "street",
    "building", "structure", "house", "residential",
    "forest", "tree", "vegetation", "crop", "farmland",
    "industrial", "factory", "warehouse",
    "airport", "port", "harbor",
    "urban", "city", "town", "village",
    "construction", "excavation",
    "bridge", "dam",
    "solar", "wind", "energy",
}

_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
_AFTER_PATTERN = re.compile(r"\bafter\s+(20\d{2})\b")
_BEFORE_PATTERN = re.compile(r"\bbefore\s+(20\d{2})\b")
_BETWEEN_PATTERN = re.compile(r"\bbetween\s+(20\d{2})\s+and\s+(20\d{2})\b")
_SINCE_PATTERN = re.compile(r"\bsince\s+(20\d{2})\b")


def _regex_parse(query: str) -> dict[str, Any]:
    text = query.lower()

    intent = "semantic_search"
    if any(w in text for w in {"change", "changed", "compare", "before", "after", "difference", "detect", "new", "appeared", "disappeared", "expansion", "widening"}):
        intent = "change_detection"
    elif any(w in text for w in {"similar", "like this", "more like", "resembles"}):
        intent = "similarity_search"

    objects = [obj for obj in _OBJECT_TERMS if obj in text]

    change_type = None
    for ctype, keywords in _CHANGE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            change_type = ctype
            break

    temporal: dict[str, str] = {}
    m = _BETWEEN_PATTERN.search(text)
    if m:
        temporal["from"] = m.group(1)
        temporal["to"] = m.group(2)
    else:
        m_after = _AFTER_PATTERN.search(text) or _SINCE_PATTERN.search(text)
        if m_after:
            temporal["after"] = m_after.group(1)
        m_before = _BEFORE_PATTERN.search(text)
        if m_before:
            temporal["before"] = m_before.group(1)

    return {
        "intent": intent,
        "objects": objects,
        "change_type": change_type,
        "temporal": temporal,
        "raw_query": query,
        "parser": "regex-fallback",
    }



_SYSTEM_PROMPT = """\
You are a satellite imagery query understanding system.
Convert the user's natural language query into a structured JSON object.
Output ONLY valid JSON. Do not add any explanation or markdown.

JSON schema:
{
  "intent": "<semantic_search|change_detection|similarity_search>",
  "objects": ["<land cover or feature type>", ...],
  "change_type": "<construction|clearance|water|road|urban_expansion|vegetation|appearance|disappearance|null>",
  "temporal": {
    "after": "<YYYY or null>",
    "before": "<YYYY or null>",
    "from": "<YYYY or null>",
    "to": "<YYYY or null>"
  },
  "raw_query": "<original query>"
}

Examples:
User: "Find images showing road construction near urban areas after 2024"
Output: {"intent":"change_detection","objects":["road","urban"],"change_type":"construction","temporal":{"after":"2024","before":null,"from":null,"to":null},"raw_query":"Find images showing road construction near urban areas after 2024"}

User: "Show rivers and waterways"
Output: {"intent":"semantic_search","objects":["river","waterway"],"change_type":null,"temporal":{"after":null,"before":null,"from":null,"to":null},"raw_query":"Show rivers and waterways"}
"""


class QueryParser:
    """
    Qwen3-powered NLP query understanding.
    
    Converts natural language to structured search parameters.
    Falls back to regex parsing when model is unavailable.
    """

    def __init__(self, model_path: str | Path, offline: bool = True):
        self.model_path = Path(model_path)
        self.offline = offline
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.backend = "regex-fallback"

        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            return
        try:
            children = list(self.model_path.iterdir())
        except (PermissionError, OSError):
            return
        if not children:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype="auto",
            )
            self.model.eval()
            self.backend = "local-qwen3"
        except Exception:
            self.tokenizer = None
            self.model = None

    def parse(self, query: str) -> dict[str, Any]:
        """
        Parse a natural language query into a structured search representation.
        
        Returns dict with: intent, objects, change_type, temporal, raw_query, parser
        """
        if self.model is not None and self.tokenizer is not None:
            result = self._parse_with_qwen(query)
            if result:
                return result

        return _regex_parse(query)

    def _parse_with_qwen(self, query: str) -> dict[str, Any] | None:
        try:
            import torch

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query.strip()},
            ]

            if hasattr(self.tokenizer, "apply_chat_template"):
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                )
            else:
                prompt = f"System: {_SYSTEM_PROMPT}\n\nUser: {query.strip()}\nOutput:"
                inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=1.0,
                )

            generated = outputs[0][inputs.shape[-1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            if "<think>" in text:
                text = text.split("</think>", 1)[-1].strip()

            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                parsed["parser"] = self.backend
                return parsed

        except Exception:
            pass
        return None

    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def describe_parse(self, parsed: dict[str, Any]) -> str:
        """Return a human-readable summary of the parsed query."""
        intent = parsed.get("intent", "semantic_search").replace("_", " ")
        objects = parsed.get("objects", [])
        change_type = parsed.get("change_type")
        temporal = parsed.get("temporal", {})
        parts = [f"Intent: **{intent}**"]
        if objects:
            parts.append(f"Objects: {', '.join(objects)}")
        if change_type:
            parts.append(f"Change type: {change_type.replace('_', ' ')}")
        t_parts = []
        if temporal.get("after"):
            t_parts.append(f"after {temporal['after']}")
        if temporal.get("before"):
            t_parts.append(f"before {temporal['before']}")
        if temporal.get("from") and temporal.get("to"):
            t_parts.append(f"between {temporal['from']} and {temporal['to']}")
        if t_parts:
            parts.append(f"Temporal: {', '.join(t_parts)}")
        return " · ".join(parts) + f" _(parser: {parsed.get('parser', 'unknown')})_"
