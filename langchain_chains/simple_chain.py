from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template = PromptTemplate(
    template="Write down the 3 find interesting facts about {topic}",
    input_variables=["topic"]
)

chain = template | model | parser
response = chain.invoke({'topic': 'Methane Gas'})
print(response)
# Output:
# Here are some interesting facts about methane gas:

# 1. **Methane: The Most Abundant Greenhouse Gas**: Methane is the second most abundant greenhouse gas in the Earth's atmosphere, after carbon dioxide. It's responsible for about 20% of the warming effect of all greenhouse gases.

# 2. **Methane Emissions from Human Activities**: The largest sources of methane emissions are agriculture (especially rice paddies and cattle), natural gas and oil systems, landfills, and coal mining. These activities release methane into the atmosphere, contributing to global warming.

# 3. **Methane: A Potent Greenhouse Gas**: Methane is 28 times more potent as a greenhouse gas than carbon dioxide over a 100-year time frame. This means that even small reductions in methane emissions can have a significant impact on the environment.

# We can visualize the how chaining works
chain.get_graph().print_ascii()
#      +-------------+       
#      | PromptInput |       
#      +-------------+       
#             *              
#             *              
#             *              
#     +----------------+     
#     | PromptTemplate |     
#     +----------------+     
#             *              
#             *              
#             *              
#    +-----------------+     
#    | ChatHuggingFace |     
#    +-----------------+     
#             *              
#             *              
#             *              
#    +-----------------+     
#    | StrOutputParser |     
#    +-----------------+     
#             *              
#             *              
#             *              
# +-----------------------+  
# | StrOutputParserOutput |  
# +-----------------------+  