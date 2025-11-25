from fastapi import FastAPI, UploadFile, File
from google.cloud import vision
from google.oauth2 import service_account
import uvicorn
import os
import json

app = FastAPI()

# ⭐ Render 환경변수(GOOGLE_CREDENTIALS)에서 JSON 문자열 불러오기
credentials_info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(credentials_info)

# Google Vision 클라이언트 생성
client = vision.ImageAnnotatorClient(credentials=credentials)

@app.get("/")
def home():
    return {"message": "ScanEat OCR Server Running!"}

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    image_content = await file.read()

    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(response.error.message)

    texts = response.text_annotations
    if not texts:
        return {"text": ""}

    extracted_text = texts[0].description
    return {"text": extracted_text}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
