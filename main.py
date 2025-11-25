from fastapi import FastAPI, UploadFile, File
from google.cloud import vision
import os
import json
import uvicorn
import tempfile

app = FastAPI()

# 🔥 Render에서 제공한 환경변수에서 JSON 키 불러오기
key_json = os.getenv("GOOGLE_CREDENTIALS_JSON")

if not key_json:
    raise Exception("환경변수 GOOGLE_CREDENTIALS_JSON 이 설정되지 않음!")

# 🔥 JSON 문자열을 임시 파일로 저장
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_key_file:
    temp_key_file.write(key_json.encode())
    temp_key_path = temp_key_file.name

# 🔥 Google API가 읽을 수 있도록 환경변수 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_path

# Vision API 클라이언트 생성
client = vision.ImageAnnotatorClient()

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
