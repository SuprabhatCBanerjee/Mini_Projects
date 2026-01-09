import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class LLMAdapter:
    """
    Epistemically constrained LLM adapter.

    This class is a STRICT boundary:
     ->It may generate text
     ->It may NOT decide belief, confidence, or status
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
        """
        Single-purpose LLM call.
        No retries, no memory, no chaining.
        """
        response = self.llm.invoke(prompt)

        # LangChain ChatOpenAI returns an AIMessage
        return response.content.strip()
