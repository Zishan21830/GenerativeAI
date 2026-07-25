from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
    )
# llm
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5
)

vectore_store = FAISS.from_documents(
    documents = all_docs,
    embedding = embeddings,
)
similarity_retriever = vectore_store.as_retriever(
    search_type="similarity", 
    search_keywords={
        'k': '1', 
        'lambda_mult': 1
        }
    )

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever = vectore_store.as_retriever(search_kwargs={'k': 3}),
    llm = model
)
query = "How to improve energy levels and maintain balance?"
similarity_results = similarity_retriever.invoke(query)
mqr_results = multi_query_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"Similarity Result: {i+1}")
    print(f"Content: {doc.page_content}...")
    
for i, doc in enumerate(mqr_results):
    print(f"Multi Query Results: {i+1}")
    print(f"Content: {doc.page_content}...")