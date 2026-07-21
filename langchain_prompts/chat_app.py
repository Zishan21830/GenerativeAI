from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt 
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

# streamlit UI Design
st.header("Research Paper Summarizer")
paper_input = st.text_input("Enter your research paper title or content")

style_input = st.selectbox(
    "How would you like to be summarize?",
    ("Beginner", "Technical", "Teacher", "Code-Oriented", "Mathematical"),
    index=None,
    placeholder="Select input style method...",
)

length_input = st.selectbox(
    "How would be the explanation style?",
    ("Short", "Medium", "Detailed"),
    index=None,
    placeholder="Select input length...",
)

template = load_prompt("Template.json")
# prompt = template.invoke(
#     {
#         "paper_input": paper_input,
#         "style_input": style_input,
#         "length_input": length_input,
#     }
# )


if st.button("Summarize"):
    chain = template | model   # Create the chain
    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input,
    })
    st.write(result.content)
