from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",  # LM Studio
    api_key="lmstudio",                   
    model="mistralai/mistral-7b-instruct-v0.3",
    temperature=0.6,
    max_tokens=300,
    max_retries=0,                        
)
