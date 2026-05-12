import logging

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api_keys import create_api_key, is_valid_api_key, list_api_keys, revoke_api_key
from app.config import Settings, get_settings
from app.groq_ine import extract_ine_with_groq, merge_ine_data
from app.ine_ocr import extract_ine_data, image_bytes_to_text, pdf_bytes_to_text
from app.schemas import IneExtractionResponse

app = FastAPI(title="INE OCR API", version="1.0.0")
logger = logging.getLogger("ine_ocr_api")

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


class CreateApiKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    has_key_protection = bool(settings.api_key) or bool(list_api_keys(settings))
    if has_key_protection and not is_valid_api_key(settings, x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida o ausente.",
        )


def require_admin_api_key(
    x_admin_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Configura ADMIN_API_KEY para usar endpoints administrativos.",
        )
    if x_admin_api_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key inválida o ausente.",
        )


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ine/extract", response_model=IneExtractionResponse)
async def extract_ine(
    file: UploadFile = File(...),
    deep_ocr: bool = Query(default=False),
    _: None = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> IneExtractionResponse:
    supported_types = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff", "application/pdf"}
    if file.content_type not in supported_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Formato no soportado. Envía una imagen JPG, PNG, WEBP, BMP, TIFF o PDF.",
        )

    file_bytes = await file.read()
    logger.info("Procesando archivo INE: filename=%s content_type=%s size=%s", file.filename, file.content_type, len(file_bytes))
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"El archivo excede el límite de {settings.max_upload_mb} MB.",
        )

    try:
        if file.content_type == "application/pdf":
            raw_text = pdf_bytes_to_text(file_bytes, settings.max_pdf_pages)
        else:
            raw_text = image_bytes_to_text(file_bytes, deep_ocr=deep_ocr)
        result = extract_ine_data(raw_text)
        should_use_groq = (
            settings.groq_api_key
            and result.confidence < 0.75
            and (
                result.validation["has_ine_keywords"]
                or result.validation["has_curp"]
                or result.validation["has_clave_elector"]
                or result.validation["has_ocr_or_cic"]
            )
        )
        if should_use_groq:
            assisted_data = extract_ine_with_groq(raw_text, settings.groq_api_key, settings.groq_model)
            merged_data = merge_ine_data(result.extracted, assisted_data)
            result = extract_ine_data(raw_text)
            result.extracted = merged_data
            result.validation["groq_assisted"] = assisted_data is not None
            if assisted_data is not None:
                result.validation["has_curp"] = result.extracted.curp is not None
                result.validation["has_clave_elector"] = result.extracted.clave_elector is not None
                result.validation["has_ocr_or_cic"] = bool(result.extracted.ocr or result.extracted.cic)
                score_keys = ["has_ine_keywords", "has_curp", "has_clave_elector", "has_ocr_or_cic"]
                result.confidence = round(sum(bool(result.validation[key]) for key in score_keys) / len(score_keys), 2)
                result.ok = result.confidence >= 0.5
                result.warnings.append("Extracción complementada con Groq a partir del texto OCR disponible.")
        return result
    except Exception as exc:
        logger.exception("Error procesando archivo INE")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No fue posible procesar el archivo: {exc}",
        ) from exc


@app.post("/api/admin/api-keys")
def create_key(
    payload: CreateApiKeyRequest,
    _: None = Depends(require_admin_api_key),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    api_key, record = create_api_key(settings, payload.name)
    return {"ok": True, "api_key": api_key, "record": record}


@app.get("/api/admin/api-keys")
def get_keys(
    _: None = Depends(require_admin_api_key),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    return {"ok": True, "api_keys": list_api_keys(settings)}


@app.delete("/api/admin/api-keys/{key_id}")
def revoke_key(
    key_id: str,
    _: None = Depends(require_admin_api_key),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    revoked = revoke_api_key(settings, key_id)
    if not revoked:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key no encontrada.")
    return {"ok": True, "revoked": True}


@app.exception_handler(HTTPException)
def http_exception_handler(_: object, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"ok": False, "detail": exc.detail})
