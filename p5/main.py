import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from model import get_model, predict

model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("starting.....")
    model_store["model"] = get_model()
    print("model loaded.")

    yield
    print("stopping service")
    model_store.clear()

app = FastAPI(lifespan=lifespan)

class PredictResponse(BaseModel):
    filename: str
    prediction: str
    response: float

@app.post("/predict", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        start_time = time.time()

        image_bytes = await file.read()
        result = predict(model=model_store["model"], image_bytes=image_bytes)

        response = (time.time() - start_time) * 1000

        return PredictResponse(
            filename=file.filename,
            prediction=result,
            response=response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message":"Application is running....."}
