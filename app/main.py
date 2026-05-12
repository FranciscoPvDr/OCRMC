from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.ine_ocr import extract_ine_data, image_bytes_to_text
from app.schemas import IneExtractionResponse

app = FastAPI(title="INE OCR API", version="1.0.0")

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida o ausente.",
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
    _: None = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> IneExtractionResponse:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Formato no soportado. Envía una imagen JPG, PNG, WEBP, BMP o TIFF.",
        )

    file_bytes = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"El archivo excede el límite de {settings.max_upload_mb} MB.",
        )

    try:
        raw_text = image_bytes_to_text(file_bytes)
        return extract_ine_data(raw_text)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No fue posible procesar la imagen: {exc}",
        ) from exc


@app.exception_handler(HTTPException)
def http_exception_handler(_: object, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"ok": False, "detail": exc.detail})
