from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a facts expert who knows fact about {animal}"),
    ("human", "Tell me {fact_count} facts.")
  ]
)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"animal": "elephant", "fact_count": 2})

print(result)