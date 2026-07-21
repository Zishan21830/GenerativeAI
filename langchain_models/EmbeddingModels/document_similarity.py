from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
docs = ["Delhi is the capital of India.",
        "Roses are red and whites",
        "Paris is the capital of France"]

query = "Delhi is beautiful"
docs_embeddings = embedding.embed_documents(docs)
query_embeddings = embedding.embed_query(query)

# In cosine similarity funciton, both the inputs should be 2D vectors
scores = cosine_similarity(docs_embeddings, [query_embeddings])
print(scores)
index, score = sorted(list(enumerate(scores)), key= lambda x: x[1])[-1]
print(query)
print(docs[index])
print(score)