from dotenv import load_dotenv
from google.cloud import firestore   
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

load_dotenv()
os.environ.pop("SSL_CERT_FILE", None)

PROJECT_ID = "langchain-c1cb7"
SESSION_ID = 'user_session_new'
COLLECTION_NAME = "chat_history"

print("Initializing Firestore client...")
client = firestore.Client(project=PROJECT_ID)

print("Initializing Firestore Chat Message History")
chat_history = FirestoreChatMessageHistory(
  session_id=SESSION_ID,
  collection=COLLECTION_NAME,
  client=client
)

print("Chat History Initialized")
print("Current chat history:", chat_history.messages)




model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
system_message = SystemMessage(content="You're an helpful history expert AI assistant. Provide reply in simple short term")


while True:
  query = input("You: ")
  if query.lower() == 'exit':
    break
  chat_history.add_user_message(HumanMessage(content=query))
  
  messages = [system_message] + chat_history.messages
  result = model.invoke(messages)
  response = result.content
  chat_history.add_ai_message(AIMessage(content=response))
  
  print(f"AI: {response}")


print("-----Message History-----")
print(chat_history)