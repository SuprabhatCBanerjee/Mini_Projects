# from langchain_core.prompts import ChatPromptTemplate
# from core.llm import llm

# prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are the Human Impact Agent. Analyze ethical, social, and long-term human effects."),
#     ("human", "{question}")
# ])

# def run_human(state):
#     response = llm.invoke(
#         prompt.format_messages(question=state["question"])
#     )
#     return {"human_view": response.content}
from langchain_core.prompts import ChatPromptTemplate
from core.llm import llm
from shared.timing import timed
prompt = ChatPromptTemplate.from_messages([
    ("human",
     """You are the Human Impact Agent.

Your job is to represent people who are NOT in the room.

Focus on:
- who is harmed silently
- power imbalances
- long-term societal consequences

Explicitly challenge efficiency- or profit-driven logic.
Assume incentives will be abused.

Be morally rigorous, not emotional.
Max 150–200 words.

Question:
{question}
""")
])

def run_human(state):
    with timed("Human Agent"):
        response = llm.invoke(
            prompt.format_messages(question=state["question"])
        )
    return {"human_view": response.content}
