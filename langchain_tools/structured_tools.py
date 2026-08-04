from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(..., description="First number to multiply", )
    b: int = Field(..., description="second number to multiply")

def multiply_nums(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a*b
    
multiply_tool = StructuredTool.from_function(
    func=multiply_nums,
    name="Multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a': 2, 'b': 3})
print(result)