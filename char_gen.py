from groq import Groq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API"))
schema = [
    ResponseSchema(name="name", description="The name of the person"),
    ResponseSchema(name="age", description="The age of the person"),
    ResponseSchema(name="characteristics", description="The person's characteristics"),
]


# Define the parser
parser = StructuredOutputParser.from_response_schemas(schema)

# Define the prompt template
template = PromptTemplate(
    input_variables=["country", "MBTI"],
    template="Give me the name, age, and characteristics of a character from this country: {country} and this MBTI: {MBTI}.\n{format_ins}",
    partial_variables={"format_ins": parser.get_format_instructions()},
)
# Define the story prompt template
story_prompt = PromptTemplate(
    input_variables=["name", "age", "characteristics"],
    template=(
        "Write a short fictional story about a person named {name}, who is {age} years old. "
        "The story should reflect the following characteristics: {characteristics}. "
        "Make it engaging, vivid, and emotionally resonant."
    )
)

# Define the full chain using RunnablePassthrough
def call_groq_llm(inputs):
    prompt_text = template.format(**inputs)
    response = client.chat.completions.create(
        model="qwen-qwq-32b",
        messages=[
            {
                "role": "system",
                "content": "You are a character generator. You will generate characters based on the country and MBTI type provided.",
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ],
        temperature=0.6,
        max_tokens=4096,
        top_p=0.95,
        stream=False,
    )
    return {"output": response.choices[0].message.content}

# Define the story generation function
def create_story(inputs):
    story_text = story_prompt.format(**inputs)
    response = client.chat.completions.create(
        model="qwen-qwq-32b",
        messages=[
            {"role": "system", "content": "You are a storyteller who writes vivid, emotional short stories add emoji too."},
            {"role": "user", "content": story_text},
        ],
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        stream=False,
    )
    return response.choices[0].message.content

# Chain composition
r= (lambda x: parser.parse(x["output"]))
chain = (
    RunnablePassthrough()
    | call_groq_llm
    | r
)

chain_2 = (chain|create_story)

# Streamlit UI
st.set_page_config(page_title="Character Generator", page_icon=":guardsman:", layout="wide")
st.title("🧙 Character Generator")
st.write("Generate a character based on Country and MBTI personality type.")

country = st.text_input("🌍 Enter Country")
mbti = st.text_input("🧠 Enter MBTI Type (e.g., INTP, ENFJ)")

if st.button("✨ Generate Character"):
    if country and mbti:
        st.info("Generating character...")
        with st.spinner("Generating..."):
            result = chain.invoke({"country": country, "MBTI": mbti})
            st.subheader("Generated Character:")
            st.write(result)
    else:
        st.warning("Please fill in both Country and MBTI Type.")

if st.button("♻️ Refresh"):
    st.session.clear()
    st.experimental_rerun()

if st.button("📖 Develop a Short Story"):
    if country and mbti:
        st.info("Generating a short story based on the character...")
        with st.spinner("Generating story..."):
            story = chain_2.invoke({"country": country, "MBTI": mbti})
            st.subheader("Generated Short Story:")
            st.write(story)
    else:
        st.warning("Please generate a character first before asking for a story.")
