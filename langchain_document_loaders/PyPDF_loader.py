from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

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

loader = PyPDFLoader('MoonshotAI.pdf')
docs = loader.load()
print(len(docs))

chain = prompt | model | parser
print(chain.invoke({'report': docs[0].page_content}))
