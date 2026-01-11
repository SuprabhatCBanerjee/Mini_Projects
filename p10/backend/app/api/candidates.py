# from fastapi import APIRouter, UploadFile
# import shutil
# import uuid
# from app.core.mongo import candidates_col

# router = APIRouter()

# @router.post("/")
# def create_candidate(name: str, job_id: str):
#     cid = f"cand_{uuid.uuid4().hex[:8]}"
#     candidates_col.insert_one({
#         "_id": cid,
#         "name": name,
#         "job_id": job_id,
#         "status": "CREATED"
#     })
#     return {"candidate_id": cid}

# @router.post("/{candidate_id}/resume")
# def upload_resume(candidate_id: str, file: UploadFile):
#     path = f"/data/resumes/{candidate_id}_{file.filename}"
#     with open(path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     candidates_col.update_one(
#         {"_id": candidate_id},
#         {"$set": {"resume_path": path}}
#     )
#     return {"status": "uploaded"}

from fastapi import APIRouter, UploadFile
import uuid
import shutil
from app.core.mongo import candidates_col
import os
router = APIRouter()


RESUME_DIR = "/data/resumes"

@router.post("/")
def create_candidate(name: str, job_id: str):
    cid = f"cand_{uuid.uuid4().hex[:8]}"
    candidates_col.insert_one({
        "_id": cid,
        "name": name,
        "job_id": job_id,
        "status": "CREATED"
    })
    return {"candidate_id": cid}


# @router.post("/{candidate_id}/resume")
# def upload_resume(candidate_id: str, file: UploadFile):
#     path = f"/data/resumes/{candidate_id}_{file.filename}"
#     with open(path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     candidates_col.update_one(
#         {"_id": candidate_id},
#         {"$set": {"resume_path": path}}
#     )
#     return {"status": "resume_uploaded"}


@router.post("/{candidate_id}/resume")
def upload_resume(candidate_id: str, file: UploadFile):
    # 🔒 Ensure directory exists
    os.makedirs(RESUME_DIR, exist_ok=True)

    path = os.path.join(
        RESUME_DIR,
        f"{candidate_id}_{file.filename}"
    )

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    candidates_col.update_one(
        {"_id": candidate_id},
        {"$set": {"resume_path": path}}
    )

    return {"status": "resume_uploaded"}