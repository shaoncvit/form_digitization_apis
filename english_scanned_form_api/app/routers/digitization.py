from fastapi import APIRouter, UploadFile, File, Form
from ..models import TemplateEnum, OcrEnum, DigitizationResponse
from ..services.digitization_service import DigitizationService

router = APIRouter()

@router.post("/digitize", response_model=DigitizationResponse)
def digitize_form(
    file: UploadFile = File(...),
    template_name: TemplateEnum = Form(...),
    ocr_name: OcrEnum = Form(...)
):
    service = DigitizationService()
    result = service.process_file(file, template_name, ocr_name)
    return {"result": result}