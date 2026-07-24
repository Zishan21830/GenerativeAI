from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5
)
prompt = PromptTemplate(
    template="Write down a summary of this {report}",
    input_variables=['report']
)
parser = StrOutputParser()

loader = WebBaseLoader("https://www.moonshot.ai/")
docs = loader.load()
print(len(docs))  #Output 1
print(docs[0].metadata) 
#Output: {'source': 'https://www.moonshot.ai/', 'title': 'Moonshot AI', 'description': 'Welcome to Moonshot AI. Our mission is to seek the optimal conversion from energy to intelligence.', 'language': 'en'}

chain = prompt | model | parser
print(chain.invoke({'report': docs[0].page_content}))
# Output:
# Here is a summary of the content:

# **Moonshot AI**

# Moonshot AI is a company that aims to achieve the optimal conversion from energy to intelligence. They have developed an AI model called Kimi, which is a flagship model capable of simplifying complex work. The Kimi model has various features, including:

# * Kimi K3: a new frontier of intelligence with 2.8T parameters, natively multimodal, and 1M-token context
# * Kimi Doodles: a collection of interactive doodles for holidays, events, and trends
# * Kimi App: available for download, with a Chrome extension and desktop versions for macOS and Windows

# **Research and Community**

# Moonshot AI has a research team that works towards achieving Artificial General Intelligence (AGI) and shares the latest research with the global open-source community. They have published recent research, including Kimi K2 and PerceptionBench.

# **Products and Services**

# Moonshot AI offers various products and services, including:

# * Kimi Open Platform
# * Kimi Code
# * Kimi Business
# * Agent and Agent Swarm
# * Hermes Agent and API

# The company also provides resources, such as multi-agent and parallel agent systems, and has a terms of service and privacy policy in place.