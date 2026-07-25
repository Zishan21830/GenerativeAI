from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang='en')
query = "Albert Einstein"
docs = retriever.invoke(query)
print(docs)