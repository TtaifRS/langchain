from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
os.environ.pop("SSL_CERT_FILE", None) 

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="qwen/qwen3-4b:free"
)

result = llm.invoke("What is the square root of 49?")
print(result.content)
