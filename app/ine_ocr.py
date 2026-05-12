import re
from io import BytesIO

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

from app.schemas import IneExtractedData, IneExtractionResponse

CURP_RE = re.compile(r"\b[A-Z][AEIOUX][A-Z]{2}\d{6}[HM][A-Z]{5}[A-Z0-9]\d\b")
CLAVE_ELECTOR_RE = re.compile(r"\b[A-Z]{6}\d{8}[HM]\d{3}\b")
OCR_RE = re.compile(r"\b\d{12,13}\b")
CIC_RE = re.compile(r"\b\d{9}\b")
SECCION_RE = re.compile(r"SECCI[O0]N\s*[:\-]?\s*(\d{3,5})")
VIGENCIA_RE = re.compile(r"VIGENCIA\s*[:\-]?\s*(\d{4}\s*[-/ ]\s*\d{4}|\d{4})")


def image_bytes_to_text(file_bytes: bytes) -> str:
    image = Image.open(BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(threshold, lang="spa", config=config)
    return normalize_text(text)


def normalize_text(text: str) -> str:
    text = text.upper()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[|_]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def extract_ine_data(raw_text: str) -> IneExtractionResponse:
    curp = first_match(CURP_RE, raw_text)
    clave_elector = first_match(CLAVE_ELECTOR_RE, raw_text)
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
    match = re.search(r"NOMBRE\s+([A-ZÑ ]{5,80}?)(?:\s+DOMICILIO|\s+CLAVE|\s+CURP|\s+FECHA|\s+SEXO)", text)
    if not match:
        return None, None, None
    parts = [part for part in match.group(1).split() if len(part) > 1]
    if len(parts) >= 3:
        return " ".join(parts[2:]), parts[0], parts[1]
    if len(parts) == 2:
        return parts[1], parts[0], None
    if len(parts) == 1:
        return parts[0], None, None
    return None, None, None


def extract_address(text: str) -> str | None:
    match = re.search(r"DOMICILIO\s+(.{10,160}?)(?:\s+CLAVE|\s+CURP|\s+FECHA|\s+SECCI[O0]N|\s+VIGENCIA)", text)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip()


def has_ine_keywords(text: str) -> bool:
    keywords = ["INSTITUTO NACIONAL ELECTORAL", "CREDENCIAL PARA VOTAR", "CLAVE DE ELECTOR", "CURP"]
    return any(keyword in text for keyword in keywords)


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
