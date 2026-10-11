"""
Legal Education Agent for plain-language concept explanations.

Explains Indian legal terms, citizen rights, legal procedures,
and principles (e.g. Bail, Anticipatory Bail, Cognizable vs Non-Cognizable,
FIR, Power of Attorney, Habeas Corpus, Limitation, Affidavit, Injunction, etc.)
in accessible, educational terms.
"""
from typing import Optional

from src.multilingual.translate import translate_text
from src.rag.pipeline import _call_llm, _LANG_CODE, MULTILINGUAL_STRATEGY
from src.rag.retriever import Retriever

EDUCATION_SYSTEM_PROMPT = """You are LexAI's Legal Education & Legal Literacy Assistant for Indian citizens.
Your mission is to demystify Indian legal concepts, terminology, rights, and court procedures for citizens who have no formal legal training.
Explain concepts clearly and accurately using Indian laws (e.g. Bharatiya Nagarik Suraksha Sanhita / CrPC, Bharatiya Nyaya Sanhita / IPC, Constitution of India, Indian Contract Act, Consumer Protection Act, etc.).
Always provide a thorough, helpful, and well-structured explanation with real-life examples and practical takeaways.
Do not decline to answer; teach and explain the legal concept thoroughly.
"""

EDUCATION_PROMPT_TEMPLATE = """Reference Excerpts from Corpus (if applicable):
{context}

Citizen Question / Concept to Explain: "{question}"

Please explain this in {language} using this clear educational format:

### 1. Simple Definition
Explain what this means in simple, everyday language that anyone can understand.

### 2. Legal Framework in India
Explain how this operates under Indian law, citing the governing Acts or codes where applicable (e.g. CrPC/BNSS, IPC/BNS, Constitution).

### 3. Key Categories & Distinctions
Explain the common types, variations, or key differences (e.g., bailable vs. non-bailable, regular vs. anticipatory bail, civil vs. criminal).

### 4. Real-World Practical Example
A relatable everyday example showing how this concept applies in real life.

### 5. Important Citizen Rights & Precautions
Key rights every citizen should know regarding this concept, and common mistakes or precautions.

### 6. Educational Disclaimer
State clearly that this is educational legal literacy information to understand your rights, not formal legal advice.
"""


class LegalEducationAgent:
    def __init__(self, pipeline=None):
        self.retriever = Retriever()

    def explain(self, question: str, language: str = "English") -> dict:
        # Search corpus for any matching local reference material
        chunks = self.retriever.search(question, top_k=3)
        relevant_chunks = [c for c in chunks if c.get("score", 0) > 0.2]
        if relevant_chunks:
            context = "\n\n---\n\n".join(
                f"[Source: {c['source']}]\n{c['text']}" for c in relevant_chunks
            )
        else:
            context = "Standard Indian jurisprudence and legal framework applies."

        use_translation = (
            language != "English" and MULTILINGUAL_STRATEGY == "translate_output"
        )

        if use_translation:
            prompt = EDUCATION_PROMPT_TEMPLATE.format(
                context=context, question=question, language="English"
            )
            raw_answer = _call_llm(EDUCATION_SYSTEM_PROMPT, prompt)
            lang_code = _LANG_CODE.get(language, "en")
            if lang_code != "en":
                answer_text = translate_text(raw_answer, lang_code)
            else:
                answer_text = raw_answer
        else:
            prompt = EDUCATION_PROMPT_TEMPLATE.format(
                context=context, question=question, language=language
            )
            answer_text = _call_llm(EDUCATION_SYSTEM_PROMPT, prompt)

        return {
            "question": question,
            "language": language,
            "answer": answer_text,
            "retrieved_sources": relevant_chunks,
            "agent": "legal_education",
            "education_note": (
                "This explanation is educational legal information, not professional legal advice."
            ),
        }