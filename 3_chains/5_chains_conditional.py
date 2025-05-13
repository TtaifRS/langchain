from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os

# load_dotenv()

# model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")
os.environ.pop("SSL_CERT_FILE", None) 

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat-v3-0324:free"
)

positive_feedback_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant"),
    ("human", "Generate a thank you comment for this positive feedback: {feedback}")
  ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant"),
    ("human", "Generate a response comment addressing this negative feedback: {feedback}")
  ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant"),
    ("human", "Generate a request comment for more details for the neutral feedback: {feedback}")
  ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant"),
    ("human", "Generate a message to escalate this feedback to a human agent: {feedback}")
  ]
)


classification_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant"),
    ("human", "Classify the sentiment of this feedback as one of the following: positive, negative, neutral, or escalate. Return only the single word (lowercase) that represents the sentiment: {feedback}")
  ]
)

branches = RunnableBranch(
  (
    lambda x: "positive" in x,
    positive_feedback_template | model | StrOutputParser()
  ),
  (lambda x: "negative" in x, 
   negative_feedback_template | model | StrOutputParser()
   ),
  (lambda x: "neutral" in x, 
   neutral_feedback_template | model | StrOutputParser()
   ),
  escalate_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches

review = "The product is terrible. It brokes after just one time use"

result = chain.invoke({"feedback": review})

print(result)