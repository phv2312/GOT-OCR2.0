from enum import Enum

from pydantic import BaseModel


class SupportedOCRType(str, Enum):
    ocr = "ocr"
    format = "format"


class Metadata(BaseModel):
    file_name: str
    ocr_type: SupportedOCRType
    time_elapsed: float


class OCRResponse(BaseModel):
    result: str
    metadata: Metadata
