def validate_assumptions(text: str):
    lines = [l.strip("-• ") for l in text.splitlines() if l.strip()]
    clean = []
    for l in lines:
        if len(l.split()) <= 12:
            clean.append(l)
    return clean[:5]


def call(self, prompt: str) -> str:
    if "status" in prompt.lower() or "confidence" in prompt.lower():
        raise RuntimeError("Epistemic violation: LLM asked to decide belief")

    response = self.llm.invoke(prompt)
    return response.content.strip()
