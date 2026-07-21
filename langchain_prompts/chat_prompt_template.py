from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate([
    ('system' , 'You are {domain} expert.'),
    ('human' , 'Explain in the term about the {topic}')
])

prompt = chat_template.invoke({
    "domain" : "Cricket",
    "topic" : "LPW"
})

result = model.invoke(prompt)
print(result.content)