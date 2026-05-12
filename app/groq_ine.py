import json

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
        clean_data = {field: clean_value(data.get(field)) for field in INE_FIELDS}
        return IneExtractedData(**clean_data)
    except Exception:
        return None


def clean_value(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().upper()
    return value or None


def merge_ine_data(base: IneExtractedData, assisted: IneExtractedData | None) -> IneExtractedData:
    if assisted is None:
        return base
    merged = base.model_dump()
    assisted_data = assisted.model_dump()
    for field, value in assisted_data.items():
        if not merged.get(field) and value:
            merged[field] = value
    return IneExtractedData(**merged)
