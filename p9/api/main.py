from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Epistemic Guardian")
app.include_router(router)
