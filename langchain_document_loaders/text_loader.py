from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5
)
prompt = PromptTemplate(
    template="Write down a 2 line summary of this {report}",
    input_variables=['report']
)
parser = StrOutputParser()

loader = TextLoader('kimi_3_summary.txt', encoding='UTF-8')
docs = loader.load()

chain = prompt | model | parser
print(chain.invoke({'report': docs[0].page_content}))
# Output: Kimi K3, a large language model developed by Moonshot AI, boasts 2.8 trillion parameters and advanced capabilities, making it a significant release in the AI field. Its impressive features have sparked interest, but also raised questions about infrastructure demands and long-term evaluation, amidst the rapidly evolving global AI landscape.