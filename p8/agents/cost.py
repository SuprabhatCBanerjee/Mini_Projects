# from langchain_core.prompts import ChatPromptTemplate
# from core.llm import llm

# prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are the Cost & Feasibility Agent. Focus on cost, complexity, and practicality."),
#     ("human", "{question}")
# ])

# def run_cost(state):
#     response = llm.invoke(
#         prompt.format_messages(question=state["question"])
#     )
#     return {"cost_view": response.content}

from langchain_core.prompts import ChatPromptTemplate
from core.llm import llm
from shared.timing import timed
prompt = ChatPromptTemplate.from_messages([
    ("human",
     """You are the Cost & Feasibility Agent.

Your job is to kill unrealistic plans.

Assume:
- budgets shrink
- timelines slip
- engineering complexity is underestimated

Call out hidden costs and fragile assumptions.
Challenge both Optimizer and Human arguments where impractical.

Be blunt and realistic.
Max 150–200 words.
Question:
{question}
""")
])

def run_cost(state):
    with timed("Cost Agent"):
        response = llm.invoke(
            prompt.format_messages(question=state["question"])
        )
    return {"cost_view": response.content}
