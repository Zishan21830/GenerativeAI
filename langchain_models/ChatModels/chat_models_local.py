from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs = dict(
        temperature=0.5,
        max_new_token =  50
    )
    )
model = ChatHuggingFace(llm=llm)
result = model.invoke("Explain the Mixture of Expert models in 2 lines")
print(result)