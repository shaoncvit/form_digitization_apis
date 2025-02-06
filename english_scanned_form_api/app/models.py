from pydantic import BaseModel
from enum import Enum

class TemplateEnum(str, Enum):
    arbors_carroll = "arbors_carroll"
    laurels_of_health = "laurels_of_health"

class OcrEnum(str, Enum):
    easyocr = "easyocr"
    tesseract = "tesseract"



class DigitizationResponse(BaseModel):
    result: dict