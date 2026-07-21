"""
Models can be requested to provide their response in a format matching a given schema. This is useful for ensuring the output can be easily parsed and used in subsequent processing. LangChain supports multiple schema types and methods for enforcing structured output.
1. TypeDict
2. Pydantic
3. JSON Schema
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

# Create model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
) # type: ignore
model = ChatHuggingFace(llm=llm)

# 1. TypedDict
#   a) Simple TypedDict

# class Person(TypedDict):
#     name: str
#     age: int
#     city: str
#     email: str

# Python’s TypedDict provides a simpler alternative to Pydantic models, ideal when you don’t need runtime validation.

# Problem: My name is Zishan. I'm 24 years old and currently living in Delhi. My email is zishan@example.com

# b) Annoted TypedDict

class Person(TypedDict):
    name:    Annotated[str, ..., "The name of the person"]
    age:     Annotated[int, ..., "The age of the person"]
    city:    Annotated[str, ..., "City where the person live"]
    email:   Annotated[str, ..., "Email address of the person"]
    skills:  Annotated[list[str], ..., "Create a list of the skills of the person"]
    hobbies: Annotated[Optional[list[str]], ..., "Create a list of hobbies if this person have only"]
    is_open: Annotated[Optional[Literal["yes", "no"]], ..., "Is this person is open for a job role"]

    
    
model_with_structure = model.with_structured_output(Person) # type: ignore

output = model_with_structure.invoke("My name is Zishan. I'm 24 years old and currently living in Delhi. My email is zishan@example.com, My skill set is Deep learning, Computer Vison and NLP. My hobbies are reading books, painting, and home gardening. I need a job.")

print(output)