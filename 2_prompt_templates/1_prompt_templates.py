from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as key strength. Keep it to 250 words max"


# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({
#   "tone": "energetic",
#   "company": "walton",
#   "position": "AI Engineer",
#   "skill": "AI"
# })

# result = llm.invoke(prompt)
# print(result.content)

messages = [
  ("system", "You are a comedian wo tells jokes about {topic}."),
  ("human", "Tell me {joke_count} jokes.")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "doctors", "joke_count": 3})


result = llm.invoke(prompt)

print(result.content)