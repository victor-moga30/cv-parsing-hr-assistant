import io
import re
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

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


EMAIL_PATTERN = re.compile(
    r"[\w.+-]+@[\w.-]+\.\w+",
    flags=re.IGNORECASE,
)

PHONE_PATTERN = re.compile(
    r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)"
)

LINKEDIN_PATTERN = re.compile(
    r"(?:https?://)?"
    r"(?:www\.)?"
    r"linkedin\.com/in/"
    r"[A-Za-z0-9_\-/%]+",
    flags=re.IGNORECASE,
)

GITHUB_PATTERN = re.compile(
    r"(?:https?://)?"
    r"(?:www\.)?"
    r"github\.com/"
    r"[A-Za-z0-9_\-]+",
    flags=re.IGNORECASE,
)


COMMON_CV_HEADINGS = {
    "about",
    "achievements",
    "certifications",
    "contact",
    "curriculum vitae",
    "education",
    "experience",
    "languages",
    "profile",
    "projects",
    "resume",
    "skills",
    "summary",
    "technical skills",
    "work experience",
}


TECHNICAL_OR_ROLE_TERMS = {
    "ai",
    "analyst",
    "backend",
    "bash",
    "c",
    "c++",
    "css",
    "database",
    "developer",
    "django",
    "docker",
    "engineer",
    "frontend",
    "git",
    "github",
    "html",
    "java",
    "javafx",
    "javascript",
    "jdbc",
    "manager",
    "mysql",
    "oop",
    "php",
    "postgresql",
    "python",
    "qt",
    "software",
    "sql",
    "testing",
}


WordTuple = Tuple[
    float,
    float,
    float,
    float,
    str,
    int,
    int,
    int,
]


def normalize_spaces(text: str) -> str:
    """
    Normalise whitespace while preserving meaningful
    line boundaries.
    """
    text = str(text).replace(
        "\x00",
        " ",
    )

    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    lines = []

    for raw_line in text.split("\n"):
        line = re.sub(
            r"[ \t\f\v]+",
            " ",
            raw_line,
        ).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def _alphabetic_tokens(
    text: str,
) -> List[str]:
    return re.findall(
        r"[^\W\d_]+",
        str(text),
        flags=re.UNICODE,
    )


def character_spacing_ratio(
    text: str,
) -> float:
    """
    Estimate whether PDF extraction produced one-character
    tokens such as:

        P y t h o n
        W o r k E x p e r i e n c e

    A normal document has relatively few isolated letters.
    """
    tokens = _alphabetic_tokens(text)

    if not tokens:
        return 1.0

    single_letter_tokens = sum(
        len(token) == 1
        for token in tokens
    )

    return (
        single_letter_tokens
        / len(tokens)
    )


def extraction_quality(
    text: str,
) -> float:
    """
    Produce a relative quality score used to choose
    between the available PDF extraction methods.
    """
    cleaned = normalize_spaces(text)

    if not cleaned:
        return float("-inf")

    tokens = _alphabetic_tokens(cleaned)

    if not tokens:
        return float("-inf")

    single_ratio = (
        character_spacing_ratio(cleaned)
    )

    normal_word_ratio = (
        sum(
            2 <= len(token) <= 24
            for token in tokens
        )
        / len(tokens)
    )

    giant_token_ratio = (
        sum(
            len(token) > 40
            for token in tokens
        )
        / len(tokens)
    )

    line_count = max(
        1,
        len(cleaned.splitlines()),
    )

    return (
        min(len(cleaned), 10000) / 1000.0
        + 18.0 * normal_word_ratio
        - 24.0 * single_ratio
        - 15.0 * giant_token_ratio
        + min(line_count, 50) / 25.0
    )


def _cleanup_reconstructed_line(
    text: str,
) -> str:
    text = re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()

    text = re.sub(
        r"\s+([,.;:!?%)\]])",
        r"\1",
        text,
    )

    text = re.sub(
        r"([([])\s+",
        r"\1",
        text,
    )

    text = re.sub(
        r"\bC\s*\+\s*\+\b",
        "C++",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bC\s*#\b",
        "C#",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bCI\s*/\s*CD\b",
        "CI/CD",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bJava\s*FX\b",
        "JavaFX",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bPostgre\s*SQL\b",
        "PostgreSQL",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def _infer_word_gap_threshold(
    tokens: Sequence[WordTuple],
) -> float:
    """
    Infer from PDF coordinates which spaces separate letters
    and which spaces separate complete words.
    """
    gaps = []
    widths = []

    for token in tokens:
        width = max(
            0.0,
            float(token[2])
            - float(token[0]),
        )

        widths.append(width)

    for previous, current in zip(
        tokens,
        tokens[1:],
    ):
        gap = (
            float(current[0])
            - float(previous[2])
        )

        if gap >= 0:
            gaps.append(gap)

    if not gaps:
        return float("inf")

    sorted_gaps = sorted(gaps)

    positive_widths = [
        width
        for width in widths
        if width > 0
    ]

    median_width = (
        statistics.median(
            positive_widths
        )
        if positive_widths
        else 5.0
    )

    if len(sorted_gaps) >= 3:
        best_jump = 0.0
        best_index = None

        for index in range(
            len(sorted_gaps) - 1
        ):
            lower = sorted_gaps[index]
            upper = sorted_gaps[
                index + 1
            ]

            jump = upper - lower

            if jump > best_jump:
                best_jump = jump
                best_index = index

        if best_index is not None:
            lower = sorted_gaps[
                best_index
            ]

            upper = sorted_gaps[
                best_index + 1
            ]

            if upper >= max(
                lower * 1.65,
                lower + 0.8,
            ):
                return (
                    lower + upper
                ) / 2.0

    median_gap = statistics.median(
        sorted_gaps
    )

    return max(
        median_gap * 1.9,
        median_width * 0.55,
        1.2,
    )


def _line_is_character_spaced(
    tokens: Sequence[WordTuple],
) -> bool:
    text_tokens = [
        str(token[4]).strip()
        for token in tokens
        if str(token[4]).strip()
    ]

    if len(text_tokens) < 3:
        return False

    alphabetic = [
        token
        for token in text_tokens
        if any(
            character.isalpha()
            for character in token
        )
    ]

    if not alphabetic:
        return False

    single_ratio = (
        sum(
            len(token) == 1
            for token in alphabetic
        )
        / len(alphabetic)
    )

    return single_ratio >= 0.60


def _reconstruct_line_from_words(
    tokens: Sequence[WordTuple],
) -> str:
    ordered_tokens = sorted(
        tokens,
        key=lambda token: (
            float(token[0]),
            float(token[1]),
        ),
    )

    ordered_tokens = [
        token
        for token in ordered_tokens
        if str(token[4]).strip()
    ]

    if not ordered_tokens:
        return ""

    if not _line_is_character_spaced(
        ordered_tokens
    ):
        return _cleanup_reconstructed_line(
            " ".join(
                str(token[4]).strip()
                for token in ordered_tokens
            )
        )

    threshold = (
        _infer_word_gap_threshold(
            ordered_tokens
        )
    )

    pieces: List[str] = []

    for index, token in enumerate(
        ordered_tokens
    ):
        value = str(
            token[4]
        ).strip()

        if index > 0:
            previous = ordered_tokens[
                index - 1
            ]

            gap = (
                float(token[0])
                - float(previous[2])
            )

            if gap > threshold:
                pieces.append(" ")

        pieces.append(value)

    return _cleanup_reconstructed_line(
        "".join(pieces)
    )


def _extract_with_fitz_words(
    file_bytes: bytes,
) -> str:
    """
    Use the geometric position of each PDF word or glyph.

    This is the most important extractor for CVs exported
    with artificial letter spacing.
    """
    if fitz is None:
        return ""

    page_texts = []

    try:
        with fitz.open(
            stream=file_bytes,
            filetype="pdf",
        ) as document:
            for page in document:
                words = (
                    page.get_text(
                        "words",
                        sort=True,
                    )
                    or []
                )

                if not words:
                    continue

                grouped: Dict[
                    Tuple[int, int],
                    List[WordTuple],
                ] = defaultdict(list)

                for word in words:
                    if len(word) < 8:
                        continue

                    word_tuple: WordTuple = (
                        float(word[0]),
                        float(word[1]),
                        float(word[2]),
                        float(word[3]),
                        str(word[4]),
                        int(word[5]),
                        int(word[6]),
                        int(word[7]),
                    )

                    line_key = (
                        word_tuple[5],
                        word_tuple[6],
                    )

                    grouped[
                        line_key
                    ].append(word_tuple)

                ordered_lines = sorted(
                    grouped.values(),
                    key=lambda line_tokens: (
                        min(
                            token[1]
                            for token
                            in line_tokens
                        ),
                        min(
                            token[0]
                            for token
                            in line_tokens
                        ),
                    ),
                )

                lines = []

                for line_tokens in ordered_lines:
                    line = (
                        _reconstruct_line_from_words(
                            line_tokens
                        )
                    )

                    if line:
                        lines.append(line)

                if lines:
                    page_texts.append(
                        "\n".join(lines)
                    )

    except Exception:
        return ""

    return normalize_spaces(
        "\n".join(page_texts)
    )


def _extract_with_fitz_text(
    file_bytes: bytes,
) -> str:
    if fitz is None:
        return ""

    page_texts = []

    try:
        with fitz.open(
            stream=file_bytes,
            filetype="pdf",
        ) as document:
            for page in document:
                page_text = (
                    page.get_text(
                        "text",
                        sort=True,
                    )
                    or ""
                )

                if page_text.strip():
                    page_texts.append(
                        page_text
                    )

    except Exception:
        return ""

    return normalize_spaces(
        "\n".join(page_texts)
    )


def _extract_with_pypdf(
    file_bytes: bytes,
    layout: bool = False,
) -> str:
    page_texts = []

    try:
        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        for page in reader.pages:
            if layout:
                try:
                    page_text = (
                        page.extract_text(
                            extraction_mode="layout"
                        )
                        or ""
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    page_text = ""

            else:
                page_text = (
                    page.extract_text()
                    or ""
                )

            if page_text.strip():
                page_texts.append(
                    page_text
                )

    except Exception:
        return ""

    return normalize_spaces(
        "\n".join(page_texts)
    )


def _extract_with_ocr(
    file_bytes: bytes,
) -> str:
    if (
        fitz is None
        or pytesseract is None
    ):
        return ""

    page_texts = []

    try:
        with fitz.open(
            stream=file_bytes,
            filetype="pdf",
        ) as document:
            for page in document:
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(
                        2.2,
                        2.2,
                    ),
                    alpha=False,
                )

                image = Image.open(
                    io.BytesIO(
                        pixmap.tobytes(
                            "png"
                        )
                    )
                )

                page_text = (
                    pytesseract
                    .image_to_string(image)
                    or ""
                )

                if page_text.strip():
                    page_texts.append(
                        page_text
                    )

    except Exception:
        return ""

    return normalize_spaces(
        "\n".join(page_texts)
    )


def extract_text_from_pdf_bytes(
    file_bytes: bytes,
) -> str:
    """
    Try several extractors and select the result with
    the best quality score.

    The old implementation returned the first extraction
    longer than 80 characters, even when every letter was
    separated. That reduced both semantic similarity and
    skill extraction accuracy.
    """
    candidates = [
        _extract_with_fitz_words(
            file_bytes
        ),
        _extract_with_pypdf(
            file_bytes,
            layout=True,
        ),
        _extract_with_fitz_text(
            file_bytes
        ),
        _extract_with_pypdf(
            file_bytes,
            layout=False,
        ),
    ]

    candidates = [
        candidate
        for candidate in candidates
        if candidate.strip()
    ]

    if not candidates:
        return _extract_with_ocr(
            file_bytes
        )

    best_text = max(
        candidates,
        key=extraction_quality,
    )

    extraction_still_bad = (
        len(best_text) < 120
        or character_spacing_ratio(
            best_text
        ) > 0.35
    )

    if extraction_still_bad:
        ocr_text = _extract_with_ocr(
            file_bytes
        )

        if (
            ocr_text
            and extraction_quality(
                ocr_text
            )
            > extraction_quality(
                best_text
            )
        ):
            best_text = ocr_text

    return normalize_spaces(
        best_text
    )


def extract_text_from_image_bytes(
    file_bytes: bytes,
) -> str:
    if pytesseract is None:
        return ""

    try:
        image = Image.open(
            io.BytesIO(file_bytes)
        )

        text = (
            pytesseract
            .image_to_string(image)
            or ""
        )

        return normalize_spaces(
            text
        )

    except Exception:
        return ""


def extract_text_from_uploaded_file(
    uploaded_file,
) -> str:
    if uploaded_file is None:
        return ""

    file_bytes = (
        uploaded_file.getvalue()
    )

    file_name = str(
        uploaded_file.name
    ).lower()

    if file_name.endswith(".pdf"):
        return (
            extract_text_from_pdf_bytes(
                file_bytes
            )
        )

    if file_name.endswith(
        (
            ".png",
            ".jpg",
            ".jpeg",
        )
    ):
        return (
            extract_text_from_image_bytes(
                file_bytes
            )
        )

    if file_name.endswith(".txt"):
        try:
            return normalize_spaces(
                file_bytes.decode(
                    "utf-8"
                )
            )

        except UnicodeDecodeError:
            return normalize_spaces(
                file_bytes.decode(
                    "latin-1",
                    errors="ignore",
                )
            )

    return ""


def _letters_and_name_punctuation(
    text: str,
) -> str:
    return "".join(
        character
        if (
            character.isalpha()
            or character in " .'-"
        )
        else " "
        for character in str(text)
    )


def _normalise_phone(
    phone: str,
) -> str:
    phone = re.sub(
        r"\s+",
        " ",
        str(phone),
    ).strip()

    digit_count = sum(
        character.isdigit()
        for character in phone
    )

    if 8 <= digit_count <= 15:
        return phone

    return "Unknown"


def _name_candidate_score(
    line: str,
    line_index: int,
    email_name_tokens: set[str],
) -> float:
    cleaned = (
        _letters_and_name_punctuation(
            line
        )
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip(" .'-")

    if not cleaned:
        return float("-inf")

    words = cleaned.split()
    lower_line = cleaned.casefold()

    lower_words = {
        word.casefold()
        for word in words
    }

    if not 2 <= len(words) <= 4:
        return float("-inf")

    if len(cleaned) > 60:
        return float("-inf")

    if lower_line in COMMON_CV_HEADINGS:
        return float("-inf")

    if any(
        heading in lower_line
        for heading
        in COMMON_CV_HEADINGS
    ):
        return float("-inf")

    if (
        lower_words
        & TECHNICAL_OR_ROLE_TERMS
    ):
        return float("-inf")

    if any(
        len(word) < 2
        for word in words
    ):
        return float("-inf")

    alpha_count = sum(
        character.isalpha()
        for character in cleaned
    )

    if alpha_count < 5:
        return float("-inf")

    uppercase_words = sum(
        word.isupper()
        for word in words
    )

    title_words = sum(
        word[:1].isupper()
        and word[1:].islower()
        for word in words
    )

    score = 0.0

    if uppercase_words == len(words):
        score += 5.0

    if title_words == len(words):
        score += 4.0

    score += (
        1.5
        if len(words) == 3
        else 1.0
    )

    score += max(
        0.0,
        2.0
        - line_index * 0.03,
    )

    if email_name_tokens:
        candidate_tokens = {
            word.casefold()
            for word in words
        }

        overlap = len(
            candidate_tokens
            & email_name_tokens
        )

        score += (
            7.0
            * overlap
            / max(
                1,
                len(email_name_tokens),
            )
        )

    return score


def extract_applicant_info(
    text: str,
) -> Dict[str, Any]:
    raw_text = normalize_spaces(
        text
    )

    lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
    ]

    compact_text = re.sub(
        r"\s+",
        " ",
        raw_text,
    ).strip()

    email_match = EMAIL_PATTERN.search(
        raw_text
    )

    phone_match = PHONE_PATTERN.search(
        raw_text
    )

    linkedin_match = (
        LINKEDIN_PATTERN.search(
            raw_text
        )
    )

    github_match = (
        GITHUB_PATTERN.search(
            raw_text
        )
    )

    email_name_tokens: set[str] = set()

    if email_match:
        email_local_part = (
            email_match
            .group(0)
            .split(
                "@",
                1,
            )[0]
        )

        email_name_tokens = {
            token.casefold()
            for token in re.findall(
                r"[^\W\d_]+",
                email_local_part,
                flags=re.UNICODE,
            )
            if len(token) >= 2
        }

    scored_candidates = []

    for line_index, line in enumerate(
        lines[:120]
    ):
        score = _name_candidate_score(
            line,
            line_index,
            email_name_tokens,
        )

        if score != float("-inf"):
            cleaned_line = (
                _letters_and_name_punctuation(
                    line
                )
            )

            cleaned_line = re.sub(
                r"\s+",
                " ",
                cleaned_line,
            ).strip(" .'-")

            scored_candidates.append(
                (
                    score,
                    cleaned_line,
                )
            )

    name = max(
        scored_candidates,
        default=(
            float("-inf"),
            "Unknown",
        ),
    )[1]

    if (
        name == "Unknown"
        and len(email_name_tokens) >= 2
    ):
        name = " ".join(
            token.title()
            for token in sorted(
                email_name_tokens
            )
        )

    phone = "Unknown"

    if phone_match:
        phone = _normalise_phone(
            phone_match.group(0)
        )

    return {
        "name": name,
        "email": (
            email_match.group(0)
            if email_match
            else "Unknown"
        ),
        "phone": phone,
        "linkedin": (
            linkedin_match.group(0)
            if linkedin_match
            else "Unknown"
        ),
        "github": (
            github_match.group(0)
            if github_match
            else "Unknown"
        ),
        "text_length": len(
            compact_text
        ),
    }