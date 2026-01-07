# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# load_dotenv()

# llm = ChatOpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=os.environ["HF_TOKEN"],
#     model="zai-org/GLM-4.5V:zai-org",
#     temperature=0.7,
#     max_tokens=512,
# )

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",  # LM Studio
    api_key="lmstudio",                   # dummy, required by client
    model="mistralai/mistral-7b-instruct-v0.3",
    temperature=0.6,
    max_tokens=300,
    max_retries=0,                        # VERY important
)
