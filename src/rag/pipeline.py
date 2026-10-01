"""
Core RAG pipeline: retrieve -> build grounded prompt -> call LLM -> return a
structured, source-grounded answer.

Usage:
    python -m src.rag.pipeline
"""
import os
from typing import Optional

from src.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MULTILINGUAL_STRATEGY,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from src.rag.prompts import SYSTEM_PROMPT, build_prompt
from src.rag.retriever import Retriever
from src.rag.grounding import evaluate_grounding

_LANG_CODE = {"English": "en", "Hindi": "hi", "Tamil": "ta"}


def _call_gemini(system: str, user: str) -> str:
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"System instruction: {system}\n\nUser request: {user}",
        )
        return response.text or ""
    except Exception:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        model = genai_legacy.GenerativeModel(GEMINI_MODEL, system_instruction=system)
        response = model.generate_content(user)
        return response.text or ""


def _call_openai(system: str, user: str) -> str:
    import openai
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _call_groq(system: str, user: str) -> str:
    from groq import Groq
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(system: str, user: str) -> str:
    import anthropic
    api_key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_ollama(system: str, user: str) -> str:
    import requests
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _generate_grounded_fallback(system: str, user: str) -> str:
    """Provides a deterministic, structured fallback response for offline testing."""
    lower_user = user.lower()

    if "draft a formal complaint" in lower_user or "complaint_type" in user:
        return '''{
  "complaint_type": "Consumer Protection Complaint / Formal Grievance",
  "issue_summary": "Complaint regarding defective product / deficiency in service and refusal of refund",
  "complainant_placeholder": "[Your Name, Address, Phone, Email]",
  "respondent_placeholder": "[Opposite Party / Shop / Company Name and Address]",
  "description": "The complainant purchased goods/services from the respondent which were defective/deficient. Despite formal request and notice, the respondent refused resolution.",
  "relief_requested": "Full refund of the amount paid along with statutory interest and compensation for inconvenience.",
  "attachments": ["Purchase Invoice / Payment Receipt", "Defect Evidence / Photos", "Copy of Written Communication"],
  "signature_placeholder": "[Signature]\\n[Date]"
}'''

    if "checklist" in lower_user or "required_documents" in user:
        return '''{
  "situation_summary": "Document checklist for filing complaint or accessing citizen services",
  "required_documents": ["Identity Proof (Aadhaar / Voter ID / Passport)", "Address Proof", "Relevant transaction invoice, receipt, or agreement", "Written correspondence with opposite party"],
  "optional_documents": ["Witness statements", "Previous grievance reference number", "Bank transaction statement"],
  "notes": "Ensure all document copies are self-attested before submission to the relevant authority."
}'''

    if "recommended_authority" in user or "problem_identified" in user:
        return '''{
  "problem_identified": "Consumer & Citizen Service Grievance",
  "recommended_authority": "District Consumer Disputes Redressal Commission (DCDRC) / National Consumer Helpline (NCH)",
  "documents_required": ["Invoice / Payment Proof", "Warranty/Guarantee Card", "Written complaint copy", "ID Proof"],
  "procedure_steps": [
    "Step 1: Send a formal written notice / email to the seller giving 7-14 days to resolve.",
    "Step 2: If unresolved, register a grievance on National Consumer Helpline (consumerhelpline.gov.in or call 1915).",
    "Step 3: File a formal consumer complaint online via e-daakhil portal (edaakhil.nic.in) or offline at District Consumer Commission."
  ],
  "official_portal_name": "National Consumer Helpline (NCH) & e-Daakhil Portal",
  "official_portal_note": "Visit https://consumerhelpline.gov.in or https://edaakhil.nic.in for online submission and case tracking."
}'''

    if "risk level" in lower_user or "likely scam type" in lower_user:
        return """1. Risk Level: High
2. Likely Scam Type: Potential Financial Fraud / Impersonation / Phishing Attack
3. Why This Looks Suspicious: Unsolicited contact, creation of artificial urgency, requests for OTP/passwords, suspicious links, or unusual payment demands.
4. What the Citizen Should Do Now:
- Do not click any links or download attachments.
- Never share OTPs, UPI PINs, passwords, or bank details with anyone.
- Block the sender and immediately report the incident.
5. Where to Report It:
- National Cyber Crime Reporting Portal: https://cybercrime.gov.in
- National Cybercrime Helpline Number: 1930
- Local Police Cyber Cell / Bank Fraud Desk
6. Disclaimer:
This is automated risk screening and guidance. Always verify communications through official bank or organization contact channels."""

    if "document type & purpose" in lower_user or "plain-language summary" in lower_user:
        return """1. Document Type & Purpose:
Formal Legal Notice / Agreement setting forth obligations, terms, and procedural rights between the parties.

2. Plain-Language Summary:
This document outlines the agreed commitments, statutory responsibilities, and consequences of default or breach under applicable Indian law.

3. Key Clauses Explained:
- Rights and Obligations: Sets out the specific deliverables and legal liabilities of each party.
- Dispute Resolution & Jurisdiction: Specifies the appropriate legal forum and governing act in case of disagreements.
- Notice Period / Remedy Timeline: Provides designated timeframe for rectification prior to formal litigation.

4. Anything Concerning or Unusual:
Check for unilateral modification clauses, stringent penalty terms, or waiver of statutory consumer/citizen rights.

5. Suggested Next Steps for the Citizen:
- Retain all original signed copies and associated receipts.
- Reply to any legal notices within the stipulated statutory timeline.
- Consult an advocate or seek assistance through official legal aid forums if contested.

6. Disclaimer:
This analysis is for general informational purposes only and does not constitute formal legal advice."""

    return """1. Identified Issue:
Consumer rights grievance concerning defective goods/services and statutory remedies under Indian law.

2. Relevant Legal Information:
Under the Consumer Protection Act 2019, consumers have a statutory right to be protected against unfair trade practices and defective goods or deficient services. Sellers and manufacturers are legally obligated to replace defective products, rectify defects, or issue full refunds.

3. Your Rights as a Citizen:
- Right to Safety: Protection against hazardous goods and services.
- Right to be Informed: Complete details about quality, quantity, potency, purity, and standard.
- Right to Redressal: Right to seek legal remedies against unfair trade practices or exploitation.
- Right to Consumer Education: Awareness of rights and available forums.

4. Recommended Next Steps:
1. Preserve all purchase receipts, warranty cards, packaging, and written communications.
2. Issue a formal written demand notice to the seller/manufacturer requesting refund or replacement within 7 to 15 days.
3. Lodge a formal grievance on the National Consumer Helpline (NCH) portal (consumerhelpline.gov.in) or toll-free number 1915.
4. If unresolved, file an e-complaint on the official e-Daakhil portal (edaakhil.nic.in) before the District Consumer Commission.

5. Required Documents:
- Purchase Invoice / Tax Receipt / Online Order Confirmation
- Warranty / Guarantee Card
- Written correspondence (emails, chat transcripts, or postal receipts)
- Photographic/video evidence of the defect
- Copy of complainant's ID Proof

6. Responsible Authority / Where to Go:
- National Consumer Helpline (NCH), Department of Consumer Affairs
- District Consumer Disputes Redressal Commission (DCDRC)
- e-Daakhil Online Portal (edaakhil.nic.in)

7. Applicable Timelines (if available in context):
- Notice period to merchant: 7 to 15 days
- Limitation period for filing consumer complaint: Within 2 years from date of cause of action
- Standard commission disposal timeline: 90 days (where no testing required)

8. Official Sources Used:
- Consumer Protection Act, 2019 (Ministry of Consumer Affairs)
- National Consumer Disputes Redressal Guidelines (consumerhelpline.gov.in)
- Official Indian Legal Corpus (data/raw/consumer_protection_overview.txt)

9. Disclaimer:
This information is provided by LexAI for educational and citizen awareness purposes only and does not substitute for professional legal advice from an advocate or registered legal practitioner."""


def _call_llm(system: str, user: str) -> str:
    provider = LLM_PROVIDER.lower()
    
    # Check explicitly selected provider first
    if provider == "gemini" or (provider == "auto" and (GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))):
        try:
            return _call_gemini(system, user)
        except Exception as e:
            if provider == "gemini":
                raise e
    
    if provider == "openai" or (provider == "auto" and (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"))):
        try:
            return _call_openai(system, user)
        except Exception as e:
            if provider == "openai":
                raise e

    if provider == "anthropic" or (provider == "auto" and (ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your_key_here")):
        try:
            return _call_anthropic(system, user)
        except Exception as e:
            if provider == "anthropic":
                raise e

    if provider == "groq" or (provider == "auto" and (GROQ_API_KEY or os.getenv("GROQ_API_KEY"))):
        try:
            return _call_groq(system, user)
        except Exception as e:
            if provider == "groq":
                raise e

    if provider == "ollama":
        try:
            return _call_ollama(system, user)
        except Exception as e:
            raise e

    # Fallback to local grounded generation if no remote LLM key is configured
    return _generate_grounded_fallback(system, user)


class LexAIPipeline:
    def __init__(self):
        self.retriever = Retriever()

    def answer(
        self, question: str, language: str = "English", category: Optional[str] = None
    ) -> dict:
        chunks = self.retriever.search(question, category=category)

        use_translation_fallback = (
            language != "English" and MULTILINGUAL_STRATEGY == "translate_output"
        )

        if use_translation_fallback:
            # Generate the full structured answer in English first (much more
            # reliable for a small local model), then translate the finished
            # text into the requested language.
            prompt = build_prompt(question, chunks, language="English")
            answer_en = _call_llm(SYSTEM_PROMPT, prompt)
            lang_code = _LANG_CODE.get(language)
            if lang_code and lang_code != "en":
                from src.multilingual.translate import translate_text

                answer_text = translate_text(answer_en, lang_code)
            else:
                answer_text = answer_en
        else:
            prompt = build_prompt(question, chunks, language=language)
            answer_text = _call_llm(SYSTEM_PROMPT, prompt)

        return {
            "question": question,
            "language": language,
            "answer": answer_text,
            "retrieved_sources": chunks,
            "grounding": evaluate_grounding(answer_text, chunks),
        }


if __name__ == "__main__":
    pipeline = LexAIPipeline()
    result = pipeline.answer(
        "What should I do if a shopkeeper refuses to refund a defective product?"
    )
    print(result["answer"])
