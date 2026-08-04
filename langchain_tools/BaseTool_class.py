from typing import type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(..., description="First number to multiply", )
    b: int = Field(..., description="second number to multiply")
    

class MultiplyTool(BaseTool):
    name: str = "Multipy"
    description: str = "Multiply"
    
    args_schema: type[BaseModel] = MultiplyInput
    
    def _run(self, a: int, b: int) -> int:
        return a+b

def multiply_nums(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a*b

multiply_tool = MultiplyTool()
result = multiply_tool.invoke({'a': 2, 'b': 3})
print(result)