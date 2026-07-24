from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5
)
prompt = PromptTemplate(
    template="Write down a summary of this smartphone {detail}",
    input_variables=['detail']
)
parser = StrOutputParser()

loader = CSVLoader(file_path="smartphones_cleaned_v6.csv")
docs = loader.load()
print(len(docs))
print(docs[0])

chain = prompt | model | parser
print(chain.invoke({'detail': docs[0]}))