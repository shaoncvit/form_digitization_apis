# app/main.py
# app/main.py
import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from .routers import digitization  # Use relative import

app = FastAPI(title="English Scanned Form Digitization API")

app.include_router(digitization.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)