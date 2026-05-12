import json
import re

import requests

from app.schemas import IneExtractedData

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
INE_FIELDS = {
    "nombre",
    "primer_apellido",
    "segundo_apellido",
    "curp",
    "clave_elector",
    "ocr",
    "cic",
    "seccion",
    "vigencia",
    "domicilio",
}
CURP_RE = re.compile(r"^[A-Z][AEIOUX][A-Z]{2}\d{6}[HM][A-Z]{5}[A-Z0-9]\d$")
CLAVE_ELECTOR_RE = re.compile(r"^[A-Z]{6}[A-Z0-9]{12}$")
OCR_RE = re.compile(r"^\d{12,13}$")
CIC_RE = re.compile(r"^\d{9}$")
SECCION_RE = re.compile(r"^\d{3,5}$")
VIGENCIA_RE = re.compile(r"^\d{4}([-/ ]\d{4})?$")


def extract_ine_with_groq(raw_text: str, api_key: str, model: str) -> IneExtractedData | None:
    if not raw_text.strip():
        return None
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un extractor de datos de credenciales mexicanas INE. "
                    "Devuelve solo JSON válido. No inventes datos. "
                    "Si un campo no aparece con claridad, usa null."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extrae estos campos desde el OCR crudo: nombre, primer_apellido, segundo_apellido, "
                    "curp, clave_elector, ocr, cic, seccion, vigencia, domicilio. "
                    "Responde con un objeto JSON plano usando exactamente esas llaves.\n\n"
                    f"OCR crudo:\n{raw_text[:6000]}"
                ),
            },
        ],
    }
    try:
        response = requests.post(
            GROQ_CHAT_COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        clean_data = {field: clean_value(field, data.get(field)) for field in INE_FIELDS}
        return IneExtractedData(**clean_data)
    except Exception:
        return None


def clean_value(field: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().upper()
    if not value or value in {"NULL", "NONE", "N/A", "NO APLICA", "NO DISPONIBLE"}:
        return None
    value = re.sub(r"\s+", " ", value)
    if field == "curp":
        value = re.sub(r"[^A-Z0-9]", "", value)
        return value if CURP_RE.match(value) else None
    if field == "clave_elector":
        value = re.sub(r"[^A-Z0-9]", "", value)
        digit_count = sum(char.isdigit() for char in value)
        return value if CLAVE_ELECTOR_RE.match(value) and digit_count >= 6 and value[14] in {"H", "M"} else None
    if field == "ocr":
        value = re.sub(r"\D", "", value)
        return value if OCR_RE.match(value) else None
    if field == "cic":
        value = re.sub(r"\D", "", value)
        return value if CIC_RE.match(value) else None
    if field == "seccion":
        value = re.sub(r"\D", "", value)
        return value if SECCION_RE.match(value) else None
    if field == "vigencia":
        value = re.sub(r"[^0-9-/ ]", "", value).strip()
        return value if VIGENCIA_RE.match(value) else None
    if field in {"nombre", "primer_apellido", "segundo_apellido"}:
        if len(value) > 45 or len(value) < 2:
            return None
        if re.search(r"[^A-ZÁÉÍÓÚÑ ]", value):
            return None
        return value
    if field == "domicilio":
        if len(value) < 10 or len(value) > 180:
            return None
        return value
    return value


def merge_ine_data(base: IneExtractedData, assisted: IneExtractedData | None) -> IneExtractedData:
    if assisted is None:
        return base
    merged = base.model_dump()
    assisted_data = assisted.model_dump()
    for field, value in assisted_data.items():
        if not merged.get(field) and value:
            merged[field] = value
    return IneExtractedData(**merged)
