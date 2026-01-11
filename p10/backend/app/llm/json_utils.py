import json
from typing import Any

def ensure_json(obj: Any) -> dict:
    """
    Normalize LLM output into a Python dict.
    Accepts:
    - dict (returned as-is)
    - JSON string
    """
    if isinstance(obj, dict):
        return obj

    if isinstance(obj, str):
        return json.loads(obj)

    raise TypeError(f"Unsupported LLM output type: {type(obj)}")
