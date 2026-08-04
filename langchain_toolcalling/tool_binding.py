from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv
load_dotenv()

@tool 
def multiply(a: int, b: int) -> int:
    """Give two integers a and b this tool returns the their products"""
    return a*b

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5
)
query=HumanMessage("Can you multiply 27 with 51")
message = [query]

llm_with_tool = model.bind_tools([multiply])
llm_result = llm_with_tool.invoke(message)
message.append(llm_result)

tool_result = multiply.invoke(llm_result.tool_calls[0])
message.append(tool_result)

final_result = model.invoke(message)
print(final_result.content)
# Output: The result of multiplying 27 by 51 is 1377.