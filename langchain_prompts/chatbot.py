from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

chat_history = [
    SystemMessage("You are a helpful AI Assitant.")
]
llm = HuggingFaceEndpoint(
    # repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print(f"AI: {result.content}")