from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")

fact_prompt_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a facts expert who knows fact about {animal}"),
    ("human", "Tell me {fact_count} facts.")
  ]
)

translation_prompt_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a translator and convert provided text into {language}"),
    ("human", "Translate the following text to {language}: {text}")
  ]
)

count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "Bangla"})

chain = fact_prompt_template | model | StrOutputParser() | prepare_for_translation | translation_prompt_template | model | StrOutputParser()

result = chain.invoke({"animal": "seahorse", "fact_count": 2})

print(result)

