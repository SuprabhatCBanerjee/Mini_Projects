# from langchain.chat_models import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# def generate_question(skill: str, difficulty: str) -> str:
#     prompt = f"""
#     Generate one technical interview question.
#     Skill: {skill}
#     Difficulty: {difficulty}
#     Ask a reasoning-based question.
#     """
#     return llm.invoke(prompt).content


from app.llm.adapter import LLMAdapter

llm = LLMAdapter()

def generate_question(skill: str, difficulty: str) -> str:
    prompt = f"""
Generate ONE technical interview question.
Skill: {skill}
Difficulty: {difficulty}
Ask a reasoning-based question.
"""
    return llm.call(prompt)
