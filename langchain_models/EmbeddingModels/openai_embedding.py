from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# # For single sentence
# embedding = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions=32)
# result = embedding.embed_query("Explain the Mixture of Expert models in 2 lines")

docs = ["Delhi is the capital of India.",
        "Roses are red and whites",
        "Paris is the capital of France"]
embedding = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions=32)
result = embedding.embed_documents(docs)

print(str(result))