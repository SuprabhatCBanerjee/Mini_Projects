from fastapi import FastAPI
from app.api import jobs, candidates, interviews, realtime
import asyncio
from app.realtime import runtime

from app.core.logger import get_logger

logger = get_logger("APP")

app = FastAPI(title="Multi-Agent Hiring Platform")


@app.on_event("startup")
async def startup_event():
    runtime.event_loop = asyncio.get_running_loop()
    logger.info("FastAPI startup complete. Event loop captured.")


app.include_router(realtime.router)
app.include_router(jobs.router, prefix="/jobs")
app.include_router(candidates.router, prefix="/candidates")
app.include_router(interviews.router, prefix="/interviews")
