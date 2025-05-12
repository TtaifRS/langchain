from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")

summary_template = ChatPromptTemplate.from_messages([
  ("system", "You are a movie critic"),
  ("human", "Provide a brief summary of the movie {movie_name}")
])

def analyze_plot(plot):
  plot_template = ChatPromptTemplate.from_messages(
    [
      ("system", "You are a movie critic"),
      ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?")
    ]
  )
  return plot_template.format_prompt(plot=plot)

def analyze_characters(characters):
  character_template = ChatPromptTemplate.from_messages(
    [
      ("system", "You are a movie critic"),
      ("human", "Analyze the characters: {characters}. What are their strengths and weeknesses")
    ]
  )
  return character_template.format_prompt(characters=characters)


def combine_verdicts(plot_analysis, character_analysis):
  return f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"

plot_branch_chain = (
  RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain = (
  RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

chain = (
  summary_template 
  | model
  | StrOutputParser()
  | RunnableParallel(branches={"plot": plot_branch_chain, "characters": character_branch_chain})
  | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
)

result = chain.invoke({"movie_name": "The Dark Knight"})

print(result)
  