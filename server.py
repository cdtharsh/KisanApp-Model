from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import load_and_preprocess_image
from inference import predict_disease
from config import CONFIDENCE_THRESHOLD

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await load_and_preprocess_image(file)
    response = predict_disease(image, CONFIDENCE_THRESHOLD)
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
    print("Server started at http://localhost:8000")