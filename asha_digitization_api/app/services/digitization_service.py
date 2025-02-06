import os
from fastapi import UploadFile
from app.models import TemplateEnum
from digitize import asha_digitized

class DigitizationService:
    def __init__(self):
        self.asha_digitization = asha_digitized()

    def process_file(self, file: UploadFile, template_name: TemplateEnum):
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
            result = self.asha_digitization.run(file_path, template_name)
            return result
        finally:
            # Your existing logic in asha_digitized will handle the unzipping and processing
            pass