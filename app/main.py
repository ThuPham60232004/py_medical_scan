from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import threading
from app.test import main
app = FastAPI()

@app.on_event("startup")
def startup_event():
    """Tự động quét và xử lý ảnh khi server khởi động"""
    thread = threading.Thread(target=main) 
    thread.start()

@app.get("/")
async def root():
    return {"message": "Image Processing API is running!"}