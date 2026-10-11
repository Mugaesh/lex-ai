"""
Scam / fraud message analysis module.

Combines a lightweight rule-based first pass (fast, offline, catches
obvious patterns) with an LLM-based second pass (nuanced reasoning +
explanation), so the system still gives a useful answer even if the LLM
call fails or is slow.
"""
import re
from typing import List

from src.rag.pipeline import _call_llm

SCAM_SYSTEM_PROMPT = """You are LexAI's Cyber Scam & Fraud Analysis Assistant for Indian citizens.
Your job is to analyze suspicious messages (SMS, WhatsApp, emails, phone threats) for fraud indicators like UPI scams, fake job offers, digital arrest, electricity bill fraud, KYC scams, lottery/gift scams, phishing links, etc.
Provide a clear, practical assessment of the message's legitimacy, specific risks, and concrete protective steps under Indian cyber safety procedures (including cybercrime.gov.in and the national cyber helpline 1930).
Always analyze the citizen's submitted message thoroughly according to the requested format.
"""

_RULES = {
    "UPI/Payment Fraud": [r"upi", r"refund.*click", r"kyc.*updat", r"otp.*shar"],
    "Fake Job Offer": [r"work from home.*earn", r"registration fee", r"joining fee"],
    "Digital Arrest Scam": [r"digital arrest", r"\bcbi\b", r"customs.*parcel", r"video call.*officer"],
    "Phishing Link": [r"bit\.ly", r"tinyurl", r"verify.*account.*link", r"suspended.*click"],
}


def _rule_based_flags(message: str) -> List[str]:
    lower = message.lower()
    flags = []
    for category, patterns in _RULES.items():
        if any(re.search(p, lower) for p in patterns):
            flags.append(category)
    return flags


def analyze_message(message: str, language: str = "English") -> dict:
    flags = _rule_based_flags(message)

    user_prompt = f"""A citizen received the following message and wants to know if it is a scam.

Message:
\"\"\"{message}\"\"\"

Rule-based pattern matches so far: {", ".join(flags) if flags else "none"}

Respond in {language} with:
1. Risk Level (Low / Medium / High)
2. Likely Scam Type (if any)
3. Why This Looks Suspicious (or why it looks legitimate)
4. What the Citizen Should Do Now (practical protective steps)
5. Where to Report It (e.g. cybercrime.gov.in, National Cybercrime Helpline 1930, Chakshu portal)
6. Disclaimer (educational assistance, not a substitute for formal law enforcement reporting)
"""
    analysis = _call_llm(SCAM_SYSTEM_PROMPT, user_prompt)
    return {"rule_based_flags": flags, "analysis": analysis}
