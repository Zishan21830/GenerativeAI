from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

# Create chat prompt template
chat_template = ChatPromptTemplate(
    [
        ("system", "You are helpful customer support agent."),
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # In general, this data come from the databases.
        ("human", "{query}"),
    ]
)

chat_history = []
# load chat history
with open("chat_history.txt", "r") as f:
    chat_history.append(f.readlines())


# create prompt
prompt = chat_template.invoke(
    {"chat_history": chat_history, "query": "Where is my refund"}
)
