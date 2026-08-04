from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5
)

# Initialize the search tool
search_tool = TavilySearch(max_results=5)

# Create the agent
agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt=(
        "You are a helpful AI assistant. "
        "Use the search tool whenever you need up-to-date information."
    )
)

# Invoke the agent
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What are the three ways to travel from Delhi to Lucknow via road?"
            }
        ]
    }
)
print(response["messages"][-1].content)
# Output:
# There are three ways to travel from Delhi to Lucknow: 

# 1. By flight: The fastest way to reach Lucknow from Delhi is by flight, which takes approximately 1 hour and costs around ₹3,600 - ₹7,000.
# 2. By train: You can also reach Lucknow from Delhi by train, which takes around 6 hours and 30 minutes and costs approximately ₹900.
# 3. By bus: Travelling by bus is another option, which takes around 10 hours and 5 minutes and costs approximately ₹550.