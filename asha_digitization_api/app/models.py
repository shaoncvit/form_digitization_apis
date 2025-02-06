from pydantic import BaseModel
from enum import Enum

class TemplateEnum(str, Enum):
    screening_proforma = "screening_proforma"
    cbac = "cbac"

class DigitizationResponse(BaseModel):
    result: dict