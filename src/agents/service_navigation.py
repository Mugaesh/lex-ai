"""
Government Service Navigation Agent.

Turns a citizen's problem into a step-by-step navigation flow: the
recommended authority, documents needed, the procedure to follow, and any
relevant official portal/helpline — grounded in the retrieved legal/
government context where possible.
"""
from src.common.json_llm import call_llm_json
from src.rag.retriever import Retriever

NAVIGATION_SYSTEM_PROMPT = """You are LexAI's Government Service Navigation Assistant for Indian citizens.
Your job is to provide clear, actionable navigation paths for citizen legal, civil, and administrative problems.
Analyze the citizen's problem, leverage the provided context, and supplement with standard Indian administrative procedures (e.g., Labour Court / Labour Commissioner, Consumer Commission / e-Daakhil, Cyber Crime Cell / 1930, Rent Authority / Civil Court, Police / Magistrate, Municipal Corporation, Sub-Registrar Office, etc.).
You must ALWAYS respond with ONLY a valid JSON object matching the requested schema. Do not return apologies, commentary, or refusal text.
"""

NAVIGATION_PROMPT = """Context from official documents (may be incomplete or supplementary):
{context}

A citizen describes this problem: "{problem}"

Respond with ONLY a JSON object (no markdown, no commentary, no code fences), in {language}, with exactly these keys:

{{
  "problem_identified": "short label for the problem",
  "recommended_authority": "the specific government body/department/authority in India to approach",
  "documents_required": ["document 1", "document 2", "document 3"],
  "procedure_steps": ["step 1", "step 2", "step 3", "step 4"],
  "official_portal_name": "name of the relevant official Indian portal or helpline (e.g., e-Daakhil, 1930 Cybercrime, National Consumer Helpline 1915, Samadhan, etc.)",
  "official_portal_note": "practical note on how to access or contact this authority/portal"
}}
"""


class ServiceNavigationAgent:
    def __init__(self):
        self.retriever = Retriever()

    def navigate(self, problem: str, language: str = "English") -> dict:
        chunks = self.retriever.search(problem)
        context = (
            "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
            or "No specific section retrieved; apply standard Indian administrative procedure."
        )
        prompt = NAVIGATION_PROMPT.format(context=context, problem=problem, language=language)
        result = call_llm_json(NAVIGATION_SYSTEM_PROMPT, prompt)
        result["_sources"] = [c["source"] for c in chunks]
        return result
