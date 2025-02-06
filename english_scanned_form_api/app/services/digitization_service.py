import os
from fastapi import UploadFile
from app.models import TemplateEnum, OcrEnum
from digitize import english_scanned_digitized

class DigitizationService:
    def __init__(self):
        self.English_Scanned_Digitized = english_scanned_digitized()

    def process_file(self, file: UploadFile, template_name: TemplateEnum, ocr_name = OcrEnum):
        # Ensure the upload_forms directory exists
        os.makedirs("upload_forms", exist_ok=True)

        # Save the zip file
        file_path = os.path.join("upload_forms", file.filename)
        try:
            # Save uploaded zip file
            with open(file_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Process the file using your existing asha_digitized class

            print(file_path)
            result = self.English_Scanned_Digitized.run(file_path, template_name, ocr_name)
            return result
        finally:
            # Your existing logic in English_Scanned_Digitized will handle the unzipping and processing
            pass