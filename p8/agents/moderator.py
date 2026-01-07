# from langchain_core.prompts import ChatPromptTemplate
# from core.llm import llm

# prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      """You are the Moderator Agent.

# Your job is to synthesize conflicting viewpoints.

# Rules:
# - Do NOT make a final decision
# - Highlight trade-offs
# - Preserve disagreement

# Structure output as:
# 1. Consensus
# 2. Conflicts
# 3. Risk Zones
# 4. Open Questions
# """),
#     ("human",
#      """Optimizer:
# {optimizer}

# Risk:
# {risk}

# Human Impact:
# {human}

# Cost:
# {cost}
# """)
# ])

# def run_moderator(state):
#     response = llm.invoke(
#         prompt.format_messages(
#             optimizer=state["optimizer_view"],
#             risk=state["risk_view"],
#             human=state["human_view"],
#             cost=state["cost_view"],
#         )
#     )
#     return {"synthesis": response.content}
from langchain_core.prompts import ChatPromptTemplate
from core.llm import llm
from shared.timing import timed
prompt = ChatPromptTemplate.from_messages([
    ("human",
     """You are the Moderator Agent.

Your job is NOT to agree.
Your job is to expose tension.

Do the following:
1. Identify where agents AGREE
2. Identify where they DIRECTLY CONFLICT
3. Identify risks with no resolution
4. List decisions that require human judgment

Do NOT recommend a final answer.

Optimizer:
{optimizer}

Risk:
{risk}

Human Impact:
{human}

Cost:
{cost}
""")
])

def run_moderator(state):
    with timed("Moderator Agent"):
        response = llm.invoke(
            prompt.format_messages(
                optimizer=state["optimizer_view"],
                risk=state["risk_view"],
                human=state["human_view"],
                cost=state["cost_view"],
            )
        )
    return {"synthesis": response.content}
