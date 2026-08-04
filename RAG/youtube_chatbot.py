from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# qjPH9njnaVU

# Step 1a: Indexing (Document Ingestion)
video_id = "qjPH9njnaVU" #only the id
transcript = ""
try:
    yt_api = YouTubeTranscriptApi()
    transcript_list = yt_api.fetch(video_id, languages=['en'])
    
    # flatten into a plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(f"Transcript: {transcript}")
    
except TranscriptsDisabled:
    print("No caption available for this video.")


# Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents(transcript)
# print(f"Chunk: {chunks}")

# model
llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

# Embedding Model
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# vector Store
vector_store = FAISS.from_documents(chunks, embedding)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 2})
retriever.invoke("What is telegram?")

prompt = PromptTemplate(
    template="""
    You are a helpful Assistant
    Write down the answersfrom the provided transcript ONLY
    If context is insufficient then simpley write You don't know.
    
    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)

question = "Is there is duscussion of russian if yes what was the discussed?"
retrieved_docs = retriever.invoke(question)

context_docs = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({
    'context': context_docs, 
    'question': question
})

answer = model.invoke(final_prompt)
print(answer.content)