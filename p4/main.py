import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import get_model, predict

model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("starting...")
    model_store["model"] = get_model()
    print("model loaded!")

    yield
    print("shutting down")
    model_store.clear()

app = FastAPI(lifespan=lifespan)

class BotResponse(BaseModel):
    output : str
    response_time : float


@app.post("/chat", response_model=BotResponse)
async def generate(data: dict):
    try:
        start_time = time.time()
        # print(data)
        response = predict(model=model_store["model"], data=data.get("text"))

        response_time = (time.time() - start_time) * 1000

        return BotResponse(
            output=response,
            response_time=round(response_time,2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/")
async def root():
    return {"message":"App is running..."}