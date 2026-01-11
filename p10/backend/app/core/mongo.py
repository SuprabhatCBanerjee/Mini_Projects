from pymongo import MongoClient
from app.core.config import settings

client = MongoClient(settings.MONGO_URI)
db = client[settings.MONGO_DB]

jobs_col = db.jobs
candidates_col = db.candidates
sessions_col = db.interview_sessions
agent_outputs_col = db.agent_outputs
technical_interviews_col = db.technical_interviews
behavioral_interviews_col = db.behavioral_interviews
final_decisions_col = db.final_decisions
human_reviews_col = db.human_reviews
question_previews_col = db.question_previews
