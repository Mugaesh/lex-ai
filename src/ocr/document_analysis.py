"""
OCR + document understanding module.

Pipeline: PDF/image -> extracted text (Tesseract, with Hindi/Tamil trained
data) -> summarization / clause explanation / risk flagging, reusing the
same grounded LLM call used by the RAG pipeline.

Tesseract is optional — if not installed, PDF text is extracted via pypdf
and images are analysed without OCR (a warning is shown to the user).

To install Tesseract for full OCR support on Windows:
    https://github.com/UB-Mannheim/tesseract/wiki
"""
from src.rag.pipeline import _call_llm
from src.rag.prompts import SYSTEM_PROMPT

VALID_DOC_TYPES = {
    "fir",
    "rental_agreement",
    "employment_contract",
    "bank_notice",
    "government_notice",
    "general",
}

# Check once at import time whether Tesseract is available
_TESSERACT_AVAILABLE = False
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    _TESSERACT_AVAILABLE = True
except Exception:
    pass

_POPPLER_AVAILABLE = False
try:
    from pdf2image import convert_from_path as _convert_from_path
    _POPPLER_AVAILABLE = True
except Exception:
    pass


def _extract_pdf_text_fallback(file_path: str) -> str:
    """Extract text from a PDF using pypdf (no Tesseract needed)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
        return "\n".join(pages_text).strip()
    except Exception as exc:
        return f"[Could not extract PDF text: {exc}]"


def extract_text(file_path: str, languages: str = "eng+hin+tam") -> str:
    """
    Extract text from a PDF or image file.

    Prefers Tesseract OCR when available. Falls back to:
      - pypdf text extraction for PDFs
      - a descriptive placeholder for images (requires Tesseract for OCR)

    languages: tesseract language codes, e.g. 'eng+hin+tam'.
    """
    lower = file_path.lower()

    if lower.endswith(".pdf"):
        if _TESSERACT_AVAILABLE and _POPPLER_AVAILABLE:
            from pdf2image import convert_from_path
            import pytesseract
            pages = convert_from_path(file_path)
            return "\n".join(
                pytesseract.image_to_string(p, lang=languages) for p in pages
            ).strip()
        # Poppler/Tesseract not available — use pypdf
        return _extract_pdf_text_fallback(file_path)

    # Image file
    if _TESSERACT_AVAILABLE:
        from PIL import Image
        import pytesseract
        return pytesseract.image_to_string(Image.open(file_path), lang=languages).strip()

    return (
        "[Image OCR requires Tesseract. Install it from "
        "https://github.com/UB-Mannheim/tesseract/wiki and restart the app.]"
    )


def analyze_document(file_path: str, doc_type: str = "general", language: str = "English") -> dict:
    """
    doc_type: one of VALID_DOC_TYPES (fir, rental_agreement,
    employment_contract, bank_notice, government_notice, general).
    """
    if doc_type not in VALID_DOC_TYPES:
        doc_type = "general"

    raw_text = extract_text(file_path)

    user_prompt = f"""Below is extracted text from a citizen-uploaded {doc_type} document. \
The text may have minor recognition errors.

Document text:
{raw_text}

Respond in {language} with:
1. Document Type & Purpose (in simple terms)
2. Plain-Language Summary
3. Key Clauses Explained (list each important clause and what it means)
4. Anything Concerning or Unusual (flag risky/unfair clauses if present)
5. Suggested Next Steps for the Citizen
6. Disclaimer (this is general information, not legal advice)
"""
    analysis = _call_llm(SYSTEM_PROMPT, user_prompt)
    return {"extracted_text": raw_text, "analysis": analysis}


def ask_about_document(document_text: str, question: str, language: str = "English") -> str:
    """
    Answers a follow-up question about a document already analyzed by
    analyze_document() (e.g. "Can my landlord increase the rent?"), grounded
    in that document's extracted text.
    """
    user_prompt = f"""Below is the text of a document the citizen uploaded earlier.

Document text:
{document_text}

The citizen now asks: "{question}"

Answer in {language}, based on the document content above. If the document \
does not contain enough information to answer, say so clearly instead of \
guessing. Keep the answer concise and end with a one-line reminder that \
this is general information, not legal advice.
"""
    return _call_llm(SYSTEM_PROMPT, user_prompt)
