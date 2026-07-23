from abc import ABC, abstractmethod
import random

class Runnable(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def invoke(input_Data):
        pass
    

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
    
class DummyPromptTemplate(Runnable):
    def __init__(self,template, input_variables):
        super().__init__()
        self.template = template
        self.input_variables = input_variables
        
    def invoke(self, input_data):
        return self.template.format(**input_data)
    
    def format(self, input_data):
        return self.template.format(**input_data)
    
class DummyStrOutParser(Runnable):
    def __init__(self):
        super().__init__()
        
    def invoke(self, input_data):
        return input_data['response']
    def parse(self, input_data):
        return input_data['response']
    
class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        super().__init__()
        self.runnable_list = runnable_list
        
    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
            
        return input_data
        
template = DummyPromptTemplate(
    template="Write a {length} poem on {topic}",
    input_variables=['length', 'topic']
)
llm = DummyLLM()
parser = DummyStrOutParser()
chain = RunnableConnector([template, llm, parser])
response = chain.invoke({'length': 'Short', 'topic': 'india'})
print(response)