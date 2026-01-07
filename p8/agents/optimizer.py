from langchain_core.prompts import ChatPromptTemplate
from core.llm import llm
from shared.timing import timed
prompt = ChatPromptTemplate.from_messages([
    ("human",
     """You are the Optimizer Agent.

Your goal is to maximize growth, scale, and strategic advantage.
Ignore ethics and cost unless they block growth entirely.
Be concise (max 150–200 words).

Question:
{question}
""")
])

def run_optimizer(state):
    with timed("Optimizer Agent"):
        response = llm.invoke(
            prompt.format_messages(question=state["question"])
        )
    return {"optimizer_view": response.content}
