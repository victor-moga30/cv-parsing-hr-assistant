import io
import re
from typing import Dict, Any

from PIL import Image
from pypdf import PdfReader

try:
    import fitz
except Exception:
    fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None


def normalize_spaces(text: str) -> str:
    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    texts = []

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                texts.append(page_text)
    except Exception:
        pass

    text = normalize_spaces(" ".join(texts))
    if len(text) >= 80:
        return text

    if fitz is None:
        return text

    texts = []

    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")

        for page in document:
            page_text = page.get_text("text") or ""
            if page_text.strip():
                texts.append(page_text)

        text = normalize_spaces(" ".join(texts))
        if len(text) >= 80:
            return text

        if pytesseract is None:
            return text

        ocr_texts = []

        for page in document:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            page_ocr = pytesseract.image_to_string(image)
            if page_ocr.strip():
                ocr_texts.append(page_ocr)

        return normalize_spaces(" ".join(ocr_texts))

    except Exception:
        return text


def extract_text_from_image_bytes(file_bytes: bytes) -> str:
    if pytesseract is None:
        return ""

    try:
        image = Image.open(io.BytesIO(file_bytes))
        return normalize_spaces(pytesseract.image_to_string(image))
    except Exception:
        return ""


def extract_text_from_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(file_bytes)

    if file_name.endswith(".png") or file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
        return extract_text_from_image_bytes(file_bytes)

    if file_name.endswith(".txt"):
        try:
            return normalize_spaces(file_bytes.decode("utf-8"))
        except Exception:
            return normalize_spaces(file_bytes.decode("latin-1", errors="ignore"))

    return ""


def extract_applicant_info(text: str) -> Dict[str, Any]:
    raw_text = str(text)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    compact_text = normalize_spaces(raw_text)

    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)

    phone_match = re.search(
        r"(\+?\d[\d\s().-]{7,}\d)",
        raw_text,
    )

    linkedin_match = re.search(
        r"(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9_\-/%]+",
        raw_text,
        flags=re.IGNORECASE,
    )

    github_match = re.search(
        r"(https?://)?(www\.)?github\.com/[A-Za-z0-9_\-]+",
        raw_text,
        flags=re.IGNORECASE,
    )

    name = "Unknown"

    forbidden = [
        "curriculum vitae",
        "resume",
        "email",
        "phone",
        "linkedin",
        "github",
        "address",
        "summary",
        "profile",
        "education",
        "experience",
        "skills",
    ]

    for line in lines[:10]:
        clean_line = re.sub(r"[^A-Za-z .'-]", " ", line).strip()
        clean_line = re.sub(r"\s+", " ", clean_line)
        lower_line = clean_line.lower()

        if len(clean_line.split()) in [2, 3] and not any(word in lower_line for word in forbidden):
            name = clean_line
            break

    return {
        "name": name,
        "email": email_match.group(0) if email_match else "Unknown",
        "phone": phone_match.group(0).strip() if phone_match else "Unknown",
        "linkedin": linkedin_match.group(0) if linkedin_match else "Unknown",
        "github": github_match.group(0) if github_match else "Unknown",
        "text_length": len(compact_text),
    }