# from langchain_core.prompts import ChatPromptTemplate
# from core.llm import llm

# prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are the Risk Guardian Agent. Identify failure modes, liabilities, and edge cases."),
#     ("human", "{question}")
# ])

# def run_risk(state):
#     response = llm.invoke(
#         prompt.format_messages(question=state["question"])
#     )
#     return {"risk_view": response.content}

from langchain_core.prompts import ChatPromptTemplate
from core.llm import llm
from shared.timing import timed
prompt = ChatPromptTemplate.from_messages([
    ("human",
     """You are the Risk Guardian Agent.

Your job is to STOP bad decisions.

Assume:
- optimistic assumptions are wrong
- edge cases will happen
- humans will misuse the system

Actively challenge the Optimizer’s likely arguments.
Highlight how this could fail publicly, legally, or catastrophically.

Be precise. Be pessimistic. No hedging.
Max 150–200 words.

Question:
{question}
""")
])

def run_risk(state):
    with timed("Risk Agent"):
        response = llm.invoke(
            prompt.format_messages(question=state["question"])
        )
    return {"risk_view": response.content}
