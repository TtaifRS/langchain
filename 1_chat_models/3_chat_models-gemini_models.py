from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()
os.environ.pop("SSL_CERT_FILE", None)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

messages = [
  SystemMessage("You are an expert in social media content strategy"),
  HumanMessage("Give a short tip to create engaging posts on Instagram"),
]

result = llm.invoke(messages)

print(result.content)
