import sys
from enum import Enum
from contextlib import asynccontextmanager
import time
from typing import AsyncIterator
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from fastapi import (
    Body,
    Depends, 
    FastAPI, 
    HTTPException, 
    Request, 
    UploadFile, 
    File
)

from webapp.dependencies import OCRModel
from webapp.schemas import (
    SupportedOCRType,
    OCRResponse,
    Metadata
)


class SupportedOCRType(str, Enum):
    ocr = "ocr"
    format = "format"


def custom_logger(quite: bool = False):
    logger.remove()
    if quite:
        return
    logger.add(
        sys.stderr,
        colorize=True,
        format="<fg #003385>[{time:MM/DD HH:mm:ss}]</> <level>{level: ^8}</level>| <level>{message}</level>",
    )


custom_logger()
pool_executor: ThreadPoolExecutor | None = None
MAX_THREADPOOL_WORKERS: int = 2


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global pool_executor

    # Trigger the cache
    OCRModel.get_pipeline()
    logger.info("Init model successfully    ")

    pool_executor = ThreadPoolExecutor(
        max_workers=MAX_THREADPOOL_WORKERS
    )

    yield

    pool_executor.shutdown()


app = FastAPI(lifespan=lifespan)


def save_upload_file(upload_file: UploadFile) -> Path:
    # https://stackoverflow.com/questions/63580229/how-to-save-uploadfile-in-fastapi
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


@app.post("/ai/infer/")
async def predict(
    request: Request,
    file: UploadFile = File(..., media_type="multipart/form-data"),
    ocr_type: str = Body(min_length=1, default="ocr"),
    ocr_model: OCRModel = Depends(OCRModel.get_pipeline)
):
    # Try parsing ocr_type
    try:
        supported_type: SupportedOCRType = SupportedOCRType(ocr_type)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown type: {ocr_type}"
        )

    # Try saving internal file path
    try:
        file_path = str(save_upload_file(file))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Can not save file input. Got error: {str(e)}"
        )

    # OCR
    try:
        time_start = time.time()
        result = ocr_model.process(file_path, supported_type.value)
        time_elapsed = time.time() - time_start
    except Exception as e:
        logger.warning(e)
        raise HTTPException(
            status_code=500,
            detail=f"Run OCR failed. Got error: {str(e)}"
        )

    # Conduct response
    try:
        response: OCRResponse = OCRResponse(
            result=result,
            metadata=Metadata(
                file_name=file.filename,
                ocr_type=ocr_type,
                time_elapsed=time_elapsed
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error while parsing response. Got error: {e}"
        )

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8881)
