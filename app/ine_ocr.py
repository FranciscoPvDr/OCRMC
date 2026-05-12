import re
from io import BytesIO

import cv2
import numpy as np
import pytesseract
import pypdfium2 as pdfium
from PIL import Image, ImageOps

from app.schemas import IneExtractedData, IneExtractionResponse

CURP_RE = re.compile(r"\b[A-Z][AEIOUX][A-Z]{2}\d{6}[HM][A-Z]{5}[A-Z0-9]\d\b")
CLAVE_ELECTOR_RE = re.compile(r"\b[A-Z]{6}\d{8}[HM]\d{3}\b")
OCR_RE = re.compile(r"\b\d{12,13}\b")
CIC_RE = re.compile(r"\b\d{9}\b")
SECCION_RE = re.compile(r"SECCI[O0]N\s*[:\-]?\s*(\d{3,5})")
VIGENCIA_RE = re.compile(r"VIGENCIA\s*[:\-]?\s*(\d{4}\s*[-/ ]\s*\d{4}|\d{4})")
INE_KEYWORDS = ["INSTITUTO NACIONAL ELECTORAL", "CREDENCIAL PARA VOTAR", "CLAVE DE ELECTOR", "CURP", "DOMICILIO", "VIGENCIA"]


def image_bytes_to_text(file_bytes: bytes, deep_ocr: bool = False) -> str:
    image = Image.open(BytesIO(file_bytes))
    if deep_ocr:
        return pil_image_to_text(image)
    return fast_pil_image_to_text(image)


def pdf_bytes_to_text(file_bytes: bytes, max_pages: int) -> str:
    document = pdfium.PdfDocument(file_bytes)
    embedded_text = extract_pdf_embedded_text(document, max_pages)
    if score_ocr_text(embedded_text) >= 30:
        document.close()
        return embedded_text
    texts = []
    for page_index in range(min(len(document), max_pages)):
        page = document[page_index]
        bitmap = page.render(scale=1.8)
        image = bitmap.to_pil()
        texts.append(fast_pil_image_to_text(image))
    document.close()
    return normalize_text("\n".join(texts))


def extract_pdf_embedded_text(document: pdfium.PdfDocument, max_pages: int) -> str:
    texts = []
    for page_index in range(min(len(document), max_pages)):
        text_page = document[page_index].get_textpage()
        texts.append(text_page.get_text_range())
    return normalize_text("\n".join(texts))


def pil_image_to_text(image: Image.Image) -> str:
    image = ImageOps.exif_transpose(image).convert("RGB")
    fast_text = fast_pil_image_to_text(image)
    if score_ocr_text(fast_text) >= 30:
        return fast_text
    best_text = ""
    best_score = -1
    for candidate in build_ocr_candidates(image, include_rotations=True):
        text = normalize_text(pytesseract.image_to_string(candidate, lang="spa+eng", config="--oem 3 --psm 6"))
        score = score_ocr_text(text)
        if score > best_score:
            best_text = text
            best_score = score
        if best_score >= 60:
            return best_text
    return best_text


def fast_pil_image_to_text(image: Image.Image) -> str:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray = resize_for_ocr(gray)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = sharpen_image(gray)
    return normalize_text(pytesseract.image_to_string(gray, lang="spa+eng", config="--oem 3 --psm 6"))


def build_ocr_candidates(image: Image.Image, include_rotations: bool) -> list[np.ndarray]:
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    candidates = []
    variants = rotate_variants(cv_image) if include_rotations else [cv_image]
    for rotated in variants:
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        gray = resize_for_ocr(gray)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        sharpened = sharpen_image(normalized)
        threshold = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        candidates.append(threshold)
    return candidates


def rotate_variants(image: np.ndarray) -> list[np.ndarray]:
    return [
        image,
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_180),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]


def resize_for_ocr(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape[:2]
    target_width = 1600
    if width >= target_width:
        return gray
    scale = target_width / width
    return cv2.resize(gray, (target_width, int(height * scale)), interpolation=cv2.INTER_CUBIC)


def sharpen_image(gray: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)


def score_ocr_text(text: str) -> int:
    score = 0
    score += sum(20 for keyword in INE_KEYWORDS if keyword in text)
    score += 30 if CURP_RE.search(text) else 0
    score += 30 if CLAVE_ELECTOR_RE.search(text) else 0
    score += 10 if OCR_RE.search(text) else 0
    score += min(len(re.findall(r"[A-ZÁÉÍÓÚÑ]{3,}", text)), 80)
    score -= len(re.findall(r"[^A-ZÁÉÍÓÚÑ0-9\s:.,/\-]", text))
    return score


def normalize_text(text: str) -> str:
    text = text.upper()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[|_]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def extract_ine_data(raw_text: str) -> IneExtractionResponse:
    curp = find_curp(raw_text)
    clave_elector = find_clave_elector(raw_text)
    ocr = find_ocr(raw_text)
    cic = find_cic(raw_text, ocr)
    seccion = grouped_match(SECCION_RE, raw_text)
    vigencia = grouped_match(VIGENCIA_RE, raw_text)
    nombre, primer_apellido, segundo_apellido = extract_name(raw_text)
    domicilio = extract_address(raw_text)

    validation = {
        "has_ine_keywords": has_ine_keywords(raw_text),
        "has_curp": curp is not None,
        "has_clave_elector": clave_elector is not None,
        "has_ocr_or_cic": bool(ocr or cic),
    }
    score = sum(validation.values()) / len(validation)
    warnings = build_warnings(validation)

    return IneExtractionResponse(
        ok=score >= 0.5,
        confidence=round(score, 2),
        extracted=IneExtractedData(
            nombre=nombre,
            primer_apellido=primer_apellido,
            segundo_apellido=segundo_apellido,
            curp=curp,
            clave_elector=clave_elector,
            ocr=ocr,
            cic=cic,
            seccion=seccion,
            vigencia=vigencia,
            domicilio=domicilio,
        ),
        validation=validation,
        warnings=warnings,
        raw_text=raw_text,
    )


def first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0) if match else None


def grouped_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def find_curp(text: str) -> str | None:
    match = CURP_RE.search(text)
    if match:
        return match.group(0)
    compact = re.sub(r"[^A-Z0-9]", "", text)
    for candidate in re.findall(r"[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d", compact):
        return candidate
    labeled = re.search(r"CURP\.?\s*([A-Z0-9]{16,20})", text)
    if labeled:
        candidate = re.sub(r"[^A-Z0-9]", "", labeled.group(1))
        if len(candidate) >= 18:
            return candidate[:18]
    return None


def find_clave_elector(text: str) -> str | None:
    match = CLAVE_ELECTOR_RE.search(text)
    if match:
        return match.group(0)
    compact = re.sub(r"[^A-Z0-9]", "", text)
    labeled = re.search(r"CLAVE(?:DE)?[A-Z]*ELECTOR([A-Z0-9]{16,22})", compact)
    if labeled:
        return labeled.group(1)[:18]
    for candidate in re.findall(r"[A-Z]{6}[A-Z0-9]{12}", compact):
        if any(char.isdigit() for char in candidate):
            return candidate
    return None


def find_ocr(text: str) -> str | None:
    if "OCR" in text:
        match = re.search(r"OCR\s*[:\-]?\s*(\d{12,13})", text)
        if match:
            return match.group(1)
    return first_match(OCR_RE, text)


def find_cic(text: str, ocr: str | None) -> str | None:
    if "CIC" in text:
        match = re.search(r"CIC\s*[:\-]?\s*(\d{9})", text)
        if match:
            return match.group(1)
    candidates = CIC_RE.findall(text)
    return next((candidate for candidate in candidates if candidate != ocr), None)


def extract_name(text: str) -> tuple[str | None, str | None, str | None]:
    mrz_name = extract_name_from_mrz(text)
    if any(mrz_name):
        return mrz_name
    match = re.search(r"NOMBRE\s+([A-ZÑ ]{5,80}?)(?:\s+DOMICILIO|\s+CLAVE|\s+CURP|\s+FECHA|\s+SEXO)", text)
    if match:
        parts = [part for part in match.group(1).split() if len(part) > 1]
    else:
        parts = extract_name_words_near_sex(text)
    if len(parts) >= 3:
        return " ".join(parts[2:]), parts[0], parts[1]
    if len(parts) == 2:
        return parts[1], parts[0], None
    if len(parts) == 1:
        return parts[0], None, None
    return None, None, None


def extract_name_from_mrz(text: str) -> tuple[str | None, str | None, str | None]:
    matches = re.findall(r"\b([A-ZÑ]+(?:<[A-ZÑ]+)*)<<([A-ZÑ<]{3,})", text)
    if not matches:
        return None, None, None
    surname_raw, names_raw = matches[-1]
    surname_parts = [clean_name_token(part) for part in surname_raw.split("<") if clean_name_token(part)]
    name_parts = [clean_name_token(part) for part in names_raw.split("<") if clean_name_token(part)]
    if not surname_parts or not name_parts:
        return None, None, None
    primer_apellido = surname_parts[0]
    segundo_apellido = " ".join(surname_parts[1:]) if len(surname_parts) > 1 else None
    nombre = " ".join(split_joined_names(name_parts))
    return nombre, primer_apellido, segundo_apellido


def extract_name_words_near_sex(text: str) -> list[str]:
    match = re.search(r"SEXO[HM]\s+([A-ZÑ ]{8,80}?)(?:\s+DOMICILIO|\s+CLAVE|\s+CURP)", text)
    if not match:
        return []
    return [part for part in match.group(1).split() if len(part) > 1 and part not in {"DEL", "DE", "LA"}]


def split_joined_names(parts: list[str]) -> list[str]:
    names = []
    known_names = ["FRANCISCO", "ISRAEL", "JOSE", "JUAN", "MARIA", "LUIS", "CARLOS", "MIGUEL", "ANGEL"]
    for part in parts:
        matched = False
        for known_name in known_names:
            if part.startswith(known_name) and part != known_name:
                names.append(known_name)
                rest = part[len(known_name):]
                if rest:
                    names.append(rest)
                matched = True
                break
        if not matched:
            names.append(part)
    return names


def clean_name_token(token: str) -> str:
    token = re.sub(r"[^A-ZÑ]", "", token)
    token = token.replace("SI", "")
    return token


def extract_address(text: str) -> str | None:
    match = re.search(r"DOMICILIO\s+(.{10,160}?)(?:\s+CLAVE|\s+CURP|\s+FECHA|\s+SECCI[O0]N|\s+VIGENCIA)", text)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip()


def has_ine_keywords(text: str) -> bool:
    return any(keyword in text for keyword in INE_KEYWORDS)


def build_warnings(validation: dict[str, bool]) -> list[str]:
    warnings = []
    if not validation["has_ine_keywords"]:
        warnings.append("No se detectaron palabras clave suficientes para confirmar que el documento sea INE.")
    if not validation["has_curp"]:
        warnings.append("No se detectó CURP con formato válido.")
    if not validation["has_clave_elector"]:
        warnings.append("No se detectó clave de elector con formato válido.")
    if not validation["has_ocr_or_cic"]:
        warnings.append("No se detectó OCR o CIC.")
    return warnings
