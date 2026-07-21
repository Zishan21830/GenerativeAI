from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
docs = ["Delhi is the capital of India.",
        "Roses are red and whites",
        "Paris is the capital of France"]
text = "Delhi is the capital of India"
# vector = embedding.embed_query(text)
vector = embedding.embed_documents(docs)

print(str(vector))