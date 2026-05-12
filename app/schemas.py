from pydantic import BaseModel, Field


class IneExtractedData(BaseModel):
    nombre: str | None = None
    primer_apellido: str | None = None
    segundo_apellido: str | None = None
    curp: str | None = None
    clave_elector: str | None = None
    ocr: str | None = None
    cic: str | None = None
    seccion: str | None = None
    vigencia: str | None = None
    domicilio: str | None = None


class IneExtractionResponse(BaseModel):
    ok: bool
    document_type: str = "INE"
    confidence: float = Field(ge=0, le=1)
    extracted: IneExtractedData
    validation: dict[str, bool]
    warnings: list[str]
    raw_text: str
