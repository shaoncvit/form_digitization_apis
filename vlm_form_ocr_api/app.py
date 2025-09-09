import os
import uuid
import json
import subprocess
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# Paths (adjust for your environment)
MODEL_PATH = "/home/vlm/form_ocr_models/checkpoint-945000"
IMAGE_SAVE_DIR = "/home/vlm/images"
RESULT_DIR = os.path.join(MODEL_PATH, "infer_result")


app = FastAPI(
    title="Form OCR VLM",
    description="API for running OCR inference on forms using VLM",
    version="1.0.0"
)

@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    try:
        # --- Step 1: Save uploaded image ---
        ext = os.path.splitext(file.filename)[1]
        if ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            return JSONResponse(
                {"status": "error", "message": f"Invalid file type: {ext}"},
                status_code=400,
            )

        image_id = str(uuid.uuid4())
        image_path = os.path.join(IMAGE_SAVE_DIR, f"{image_id}{ext.lower()}")

        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # --- Step 2: Create temporary .jsonl file ---
        input_jsonl = f"/tmp/{image_id}.jsonl"
        query_obj = {
            "query": f"<image>{image_path}",
            "response": "demo purpose",
            "images": [image_path],
        }

        with open(input_jsonl, "w") as f:
            f.write(json.dumps(query_obj) + "\n")

        # --- Step 3: Run inference command ---
        command = [
            "swift", "infer",
            "--adapters", MODEL_PATH,
            "--stream", "true",
            "--temperature", "0",
            "--val_dataset", input_jsonl,
            "--max_new_tokens", "2048"
        ]

        subprocess.run(
            "CUDA_VISIBLE_DEVICES=0 " + " ".join(command),
            shell=True,
            check=True
        )

        # --- Step 4: Find latest result file ---
        result_files = sorted(
            [os.path.join(RESULT_DIR, f) for f in os.listdir(RESULT_DIR) if f.endswith(".jsonl")],
            key=os.path.getmtime,
            reverse=True
        )

        if not result_files:
            return JSONResponse(
                {"status": "error", "message": "No result file found"},
                status_code=500,
            )

        latest_result = result_files[0]

        # --- Step 5: Extract "response" from result ---
        with open(latest_result, "r") as f:
            first_line = f.readline()
            result_obj = json.loads(first_line)

        response_text = result_obj.get("response", "")

        # --- Step 6: Cleanup ---
        os.remove(input_jsonl)
        # optional: os.remove(image_path)

        # --- Final Response ---
        return {
            "status": "success",
            "response": response_text,
            "pretty_response": response_text.splitlines(),
            "formatted_response": response_text.replace("\n", "<br>")
        }

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500,
        )
    
if __name__ == "__main__":
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
