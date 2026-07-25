from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


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

# Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
    )

# Create vector store
vector_store = Chroma(
    collection_name="cs_defs",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

# add documents
docs = [doc1, doc2, doc3, doc4, doc5]
vector_store.add_documents(docs)

# get data
# vector_store.get(include=['embeddings', 'documents', 'metadatas'])

# Search documents
# fetch_result = vector_store.similarity_search(
#     query="positional encoding",
#     k=1
# )
# print(fetch_result)

# Output:
# [Document(id='a695642f-3ca9-4fa4-954f-a60e539a25e4', metadata={'topic': 'Rotary Positional Embeddings (RoPE)'}, page_content='\nRotary Positional Embeddings (RoPE) encode positional information by rotating\nquery and key vectors in the Transformer attention mechanism. Unlike absolute\nposition embeddings, RoPE naturally captures relative positional relationships\nand enables better extrapolation to longer sequence lengths. It has become a\nstandard positional encoding technique in many modern large language models,\nincluding LLaMA, Qwen, and DeepSeek.\n')]

# Search with similarity scores
result_score = vector_store.similarity_search_with_score(
    query="positional encoding",
    k=1, 
    # filter={'topic': 'attention'} #meta-data filtering
)
print(result_score)

# Update document
# vector_store.update_document(document_id="", document="")

# delete document
# vector_store.delete_document(ids=[id1, id2, id3, ...], document="")