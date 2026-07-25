from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="LangChain simplifies building LLM applications.", metadata={"source": "docs"}),
    Document(page_content="FAISS is an efficient library for dense vector similarity search.", metadata={"source": "meta"}),
    Document(page_content="Python is a popular programming language for AI.", metadata={"source": "general"})
]

# Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
    )

vectore_store = FAISS.from_documents(
    documents = docs,
    embedding = embeddings,
)
retriever = vectore_store.as_retriever(
    search_type="mmr", 
    search_keywords={
        'k': '1', 
        'lambda_mult': 1
        }
    )

query = "What is langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Result: {i+1}")
    print(f"Content: {doc.page_content}...")