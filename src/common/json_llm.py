"""
Shared helper for agents that need the LLM to return structured JSON
instead of free-form markdown (Complaint Drafting, Service Navigation, Checklist).
"""
import json
import re

from src.rag.pipeline import _call_llm


def extract_json_object(text: str) -> dict:
    """Extract and parse a JSON object from raw LLM output."""
    if not text:
        raise ValueError("Empty text")

    # Remove thinking/reasoning tags if present
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    # Look for markdown code fence first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            cleaned = candidate

    # Find the outermost { ... }
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidate = cleaned[brace_start : brace_end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            # Try removing trailing commas
            candidate_fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
            return json.loads(candidate_fixed)

    return json.loads(cleaned)


def call_llm_json(system: str, user: str) -> dict:
    """
    Calls the LLM and attempts to parse its response as JSON.
    Strips accidental markdown code fences, think tags, and preamble.
    On parse failure, returns the raw text under "_raw" with "_parse_error": True.
    """
    raw = _call_llm(system, user)
    try:
        return extract_json_object(raw)
    except Exception:
        return {"_raw": raw, "_parse_error": True}
