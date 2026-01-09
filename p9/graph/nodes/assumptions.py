from llm.adapter import LLMAdapter
from llm.prompts import ASSUMPTION_PROMPT
from llm.validators import validate_assumptions

llm = LLMAdapter()

def assumptions_node(state):
    raw = llm.call(ASSUMPTION_PROMPT.format(claim=state["normalized_claim"]))
    state["assumptions"] = validate_assumptions(raw)
    return state
