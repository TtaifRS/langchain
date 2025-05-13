import os 
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from gemini_image import generate_image




load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

animal_name = "Penguin"
FACT_COUNT = 2  

base_dir = Path(__file__).parent
output_dir = base_dir / "output" / "img" / animal_name.lower()

try:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"No write permission for directory: {output_dir}")
except Exception as e:
    print(f"❌ Error creating output directory {output_dir}: {str(e)}")
    exit(1)

# Prompt to generate fun facts
fact_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a facts expert who knows fun and engaging facts about {animal_name}."),
    ("human", f"Generate {FACT_COUNT} unique, short, fun facts about {{animal_name}} suitable for kids. "
              "Each fact should be distinct, engaging, and focus on interesting traits or behaviors. "
              "Return the facts as a numbered list (1. to {FACT_COUNT}.)")
])

# Prompt to turn a fact into an image prompt
img_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at creating vivid image generation prompts for kid-friendly illustrations."),
    ("human", "Based on this animal fact: \"{fact}\", "
              "create an image generation prompt for a kid-friendly illustration of a {animal_name}. "
              "The image must: "
              "- Use a vibrant, cartoonish style with bright colors and bold outlines. "
              "- Show the {animal_name} with consistent features (e.g., sleek black-and-white feathers, orange beak for penguins). "
              "- Set the scene in a cheerful arctic environment with snowy hills, icy slopes, or sparkling blue water. "
              "- Highlight the fact’s action or trait in a playful way, making the {animal_name} the focus. "
              "- Add kid-friendly details like sparkles, bubbles, or friendly expressions. "
              "Return only the image prompt text, followed by a 'Key elements' section listing how the prompt meets these requirements.")
])

# Parse numbered facts
def parse_fact_variations(variations_text):
    lines = variations_text.strip().split("\n")
    facts = [line.split(".", 1)[1].strip()
             for line in lines if line.strip().startswith(tuple(str(i) for i in range(1, FACT_COUNT + 1)))]
    return facts[:FACT_COUNT]

# Create chains
fact_chain = fact_prompt_template | model | StrOutputParser()
image_prompt_chain = img_prompt_template | model | StrOutputParser()

print(f"🚀 Generating {FACT_COUNT} facts and image prompts for {animal_name}")

# Generate the fun facts
fact_result = fact_chain.invoke({"FACT_COUNT": FACT_COUNT ,"animal_name": animal_name,})
fact_variation = parse_fact_variations(fact_result)

# Loop through each fact
for i, fact in enumerate(fact_variation):
    print(f"\n🔢 Fact #{i+1}")
    print(f"🧠 FACT: {fact}")

    # Generate image prompt for the fact
    image_prompt = image_prompt_chain.invoke({"fact": fact, "animal_name": animal_name})
    print(f"🎨 PROMPT: {image_prompt}")

    # Generate and save image
    output_path = output_dir / f"{animal_name.lower()}_{i+1}.png"
    generate_image(image_prompt, str(output_path))
