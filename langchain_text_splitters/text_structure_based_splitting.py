"""
Text is naturally organized into hierarchical units such as paragraphs, sentences, and words. We can leverage this inherent structure to inform our splitting strategy, creating split that maintain natural language flow, maintain semantic coherence within split, and adapts to varying levels of text granularity. 

LangChain’s RecursiveCharacterTextSplitter implements this concept:
1. The RecursiveCharacterTextSplitter attempts to keep larger units (e.g., paragraphs) intact.
2. If a unit exceeds the chunk size, it moves to the next level (e.g., sentences).
3. This process continues down to the word level if necessary.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader(file_path='kimi_3_summary.txt')
docs = loader.load()
print(docs)

splitter = RecursiveCharacterTextSplitter(
    # encoding_name="cl100k_base", 
    chunk_size=50, 
    chunk_overlap=0
)
print(splitter.split_text(docs))