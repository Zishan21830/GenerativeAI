from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader

text = """
class DummyLLM(Runnable):
    def __init__(self):
        super().__init__()
        print('LLM is created.')
        
    def invoke(self, prompt):
        response_list = [
            'AI stands for Artificial Intelligence',
            'Delhi is the capital of India',
            'Dakar is the capital of Senegal'
        ] 
        return {'response' : random.choice(response_list)}
    def predict(self, prompt):
        response_list = [
            'AI stands for Artificial Intelligence',
            'Delhi is the capital of India',
            'Dakar is the capital of Senegal'
        ] 
        return {'response' : random.choice(response_list)}
"""


splitter = RecursiveCharacterTextSplitter.from_language(
    # encoding_name="cl100k_base", 
    language=Language.PYTHON,
    chunk_size=100, 
    chunk_overlap=0
)
print(splitter.split_text(text))