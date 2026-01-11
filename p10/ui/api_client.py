import requests

API_BASE = "http://backend:8000"


def create_job(job_data: dict):
    return requests.post(f"{API_BASE}/jobs/", json=job_data)


def create_candidate(name: str, job_id: str):
    return requests.post(
        f"{API_BASE}/candidates/",
        params={"name": name, "job_id": job_id},
    )


def upload_resume(candidate_id: str, file):
    files = {"file": file}
    return requests.post(
        f"{API_BASE}/candidates/{candidate_id}/resume",
        files=files,
    )


def start_interview(candidate_id: str):
    return requests.post(
        f"{API_BASE}/interviews/start/{candidate_id}"
    )


def submit_answer(candidate_id: str, answer: str):
    return requests.post(
        f"{API_BASE}/interviews/answer/{candidate_id}",
        json={"answer": answer},
    )


def approve_question(candidate_id: str):
    return requests.post(
        f"{API_BASE}/interviews/approve_question/{candidate_id}"
    )
