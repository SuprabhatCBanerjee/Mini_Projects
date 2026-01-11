from fastapi import APIRouter
from app.core.mongo import jobs_col

router = APIRouter()


# @router.post("/")
# def create_job(job: dict):
#     jobs_col.insert_one(job)
#     return {"status": "job_created"}
@router.post("/")
def create_job(job: dict):
    job_doc = {
        "_id": job["job_id"],   # 🔑 CRITICAL
        "title": job["title"],
        "requirements": job["requirements"]
    }

    jobs_col.insert_one(job_doc)
    return {"status": "job_created"}

