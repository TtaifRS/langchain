from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a facts expert who knows fact about {animal}"),
    ("human", "Tell me {fact_count} facts.")
  ]
)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke({"animal": "Zebra", "fact_count": 3})

print(response)