import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class LLMAdapter:
    """
    Epistemically constrained LLM adapter.

    - Generates text only
    - No decisions
    - No memory
    - No retries
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
            model="zai-org/GLM-4.5V:zai-org",
            temperature=0.7,
            max_tokens=512,
        )

    def call(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content.strip()
