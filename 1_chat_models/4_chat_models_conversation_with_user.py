from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()
os.environ.pop("SSL_CERT_FILE", None)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

chat_history = []

system_message = SystemMessage(content="You're an helpful history expert AI assistant. Provide reply in simple short term")
chat_history.append(system_message)

while True:
  query = input("You: ")
  if query.lower() == 'exit':
    break
  chat_history.append(HumanMessage(content=query))
  
  result = model.invoke(chat_history)
  response = result.content
  chat_history.append(AIMessage(content=response))
  
  print(f"AI: {response}")


print("-----Message History-----")
print(chat_history)