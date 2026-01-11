# from langchain.chat_models import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

# def generate_scenario(role: str) -> str:
#     prompt = f"""
#     Generate one realistic behavioral interview scenario
#     for a {role}. Focus on ambiguity and responsibility.
#     """
#     return llm.invoke(prompt).content


from app.llm.adapter import LLMAdapter

llm = LLMAdapter()

def generate_scenario(role: str) -> str:
    prompt = f"""
    Generate one realistic behavioral interview scenario
    for a {role}. Focus on ambiguity and responsibility.
    """
    return llm.call(prompt)
