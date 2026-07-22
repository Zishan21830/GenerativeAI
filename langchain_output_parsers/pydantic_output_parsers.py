from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    task="text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Name of the Person")
    age: int = Field(gt=18, description="Age of the Person")
    city: str = Field(description="City of the Person")
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="GIve the name, age, and city of the fictional {place} person \n {format_instructions}",
    input_variables=["palce"],
    partial_variables={
        'format_instructions': parser.get_format_instructions()
    }
)
print(template.invoke({'place': 'indian'}))
# Output: text='GIve the name, age, and city of the fictional indian person \n The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"name": {"description": "Name of the Person", "title": "Name", "type": "string"}, "age": {"description": "Age of the Person", "exclusiveMinimum": 18, "title": "Age", "type": "integer"}, "city": {"description": "City of the Person", "title": "City", "type": "string"}}, "required": ["name", "age", "city"]}\n```'
print("\n")

chain = template | model | parser
response = chain.invoke({'place' : 'indian'})
print(response)
# output: name='Rohan Desai' age=25 city='Mumbai'

print(response.name)
# Output: Rohan Desai