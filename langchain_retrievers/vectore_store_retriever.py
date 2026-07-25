from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import WikipediaRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
doc1 = Document(
    page_content="""
Mixture of Experts (MoE) is a neural network architecture that improves model
capacity and computational efficiency by activating only a subset of specialized
sub-networks, called experts, for each input token. A routing network determines
which experts should process a given token. This sparse activation allows MoE
models to scale to hundreds of billions or even trillions of parameters while
keeping the computation per token relatively constant. Popular MoE models
include Mixtral, DeepSeek-V3, and Google's Switch Transformer.
""",
    metadata={"topic": "Mixture of Experts (MoE)"}
)

doc2 = Document(
    page_content="""
Delta Attention is an efficient attention mechanism that reduces computational
and memory overhead by focusing on the changes (deltas) between consecutive
hidden states or attention contexts instead of recomputing full attention.
It is particularly useful for long-context language models and streaming
inference, where much of the information remains unchanged across decoding
steps. Delta Attention improves inference speed while maintaining comparable
model performance.
""",
    metadata={"topic": "Delta Attention"}
)

doc3 = Document(
    page_content="""
FlashAttention is a memory-efficient implementation of the Transformer attention
algorithm. Instead of materializing the entire attention matrix, it computes
attention using tiled matrix operations that maximize GPU utilization and reduce
high-bandwidth memory accesses. FlashAttention significantly speeds up training
and inference for large language models while producing mathematically exact
attention results.
""",
    metadata={"topic": "FlashAttention"}
)

doc4 = Document(
    page_content="""
Rotary Positional Embeddings (RoPE) encode positional information by rotating
query and key vectors in the Transformer attention mechanism. Unlike absolute
position embeddings, RoPE naturally captures relative positional relationships
and enables better extrapolation to longer sequence lengths. It has become a
standard positional encoding technique in many modern large language models,
including LLaMA, Qwen, and DeepSeek.
""",
    metadata={"topic": "Rotary Positional Embeddings (RoPE)"}
)

doc5 = Document(
    page_content="""
Grouped Query Attention (GQA) is an optimization of multi-head attention where
multiple query heads share a smaller number of key and value heads. This design
reduces memory usage and speeds up inference while maintaining performance close
to standard multi-head attention. GQA is widely adopted in recent large language
models because it offers an effective trade-off between efficiency and accuracy.
""",
    metadata={"topic": "Grouped Query Attention (GQA)"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

# Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
    )

vectore_store = Chroma.from_documents(
    documents = docs,
    embedding = embeddings,
    collection_name = 'my_collection'
)
retriever = vectore_store.as_retriever(search_keywords={'k': 2})

query = "Flash Attention"
result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"Result: {i+1}")
    print(f"Content: {doc.page_content}...")