from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader(file_path='kimi_3_summary.txt')
docs = loader.load()
print(docs)

splitter = CharacterTextSplitter(
    # encoding_name="cl100k_base", 
    chunk_size=100, 
    chunk_overlap=0,
    separator=''
)
print(splitter.split_documents(docs))