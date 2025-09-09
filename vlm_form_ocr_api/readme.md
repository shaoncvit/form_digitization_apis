# Form OCR VLM API

A FastAPI-based service for running OCR inference on form images using a Vision-Language Model (VLM).  
This API allows you to upload an image (JPG, PNG, etc.), automatically runs the OCR model, and returns structured JSON output.

---

## 🚀 Features
- Upload an image (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`)
- Automatically creates a temporary `.jsonl` file for inference
- Runs the **Form OCR VLM model** via `swift infer`
- Returns JSON with:
  - **response** → raw OCR text (with `\n`)
  - **pretty_response** → line-by-line array
  - **formatted_response** → HTML-friendly (`<br>` for display)

---
## 🌐 Accessing the API

Once the server is running:

- Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc → [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Use curl

curl -X POST "http://localhost:8000/infer" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"


