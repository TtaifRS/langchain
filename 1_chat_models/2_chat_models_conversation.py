from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
os.environ.pop("SSL_CERT_FILE", None)

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat-v3-0324:free"
    # model="qwen/qwen3-4b:free"
)

messages = [
  SystemMessage("You are an expert in social media content strategy"),
  HumanMessage("Give a short tip to create engaging posts on Instagram"),
]

result = llm.invoke(messages)

print(result.content)

