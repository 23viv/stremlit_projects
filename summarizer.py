import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from groq import Groq
from dotenv import load_dotenv
import uuid
import os

load_dotenv()
groq_key = os.getenv("GROQ_API")
client = Groq(api_key=groq_key)

# Template 1: General cool assistant response
template_1 = PromptTemplate(
    input_variables=["prompt"],
    template='Yo bro, here’s the deal with your request: "{prompt}"',
)

# Template 2: Summarizing a topic
template_2 = PromptTemplate(
    input_variables=["topic"],
    template='Give a concise summary of the topic "{topic}" in a cool, point-by-point format. Keep it short, snappy, and max 3 key points.'
)

# Output parser
output_parser = StrOutputParser()

# Create chain for general assistant response
general_chain = (
    {"prompt": RunnablePassthrough()}
    | template_1
    | client.chat.completions.create(
        model="qwen-qwq-32b",
        messages=[
            {
                "role": "system",
                "content": "Act as a cool assistant with point-by-point info."
            },
            {
                "role": "user",
                "content": "{prompt}"
            }
        ],
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=False
    )
    | output_parser
)

# Create chain for summarizing a topic
summary_chain = (
    {"topic": RunnablePassthrough()}
    | template_2
    | client.chat.completions.create(
        model="qwen-qwq-32b",
        messages=[
            {
                "role": "system",
                "content": "Act as a cool assistant summarizing topics in a point-by-point format."
            },
            {
                "role": "user",
                "content": "{topic}"
            }
        ],
        temperature=0.6,
        max_completion_tokens=512,
        top_p=0.95,
        stream=False
    )
    | output_parser
)

# Combined chain for both general and summary responses
combined_chain = RunnableParallel(
    general_response=general_chain,
    summary_response=summary_chain
)

st.set_page_config(page_title="Cool Assistant", page_icon="🤖", layout="centered")
st.title("Mrs. Cool Assistant 😎")
st.markdown("Get blazing-fast responses from a cool AI assistant 😎")

user_input = st.text_input("💬 You:", key="user_input")
if user_input:
    st.markdown("**🤖 Bot:**")
    with st.spinner("YOY I AM COOKING..."):
        # Invoke combined chain
        result = combined_chain.invoke(user_input)
        
        # Display general response
        st.markdown("**General Response:**")
        st.write(result["general_response"])
        
        # Display summary response
        st.markdown("**Topic Summary:**")
        st.write(result["summary_response"])