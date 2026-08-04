from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a*b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a+b

@tool
def power(n: int, pow: int):
    """Calculate the power"""
    return n**pow

class MathToolKit:
    def get_tools(self):
        return [add, multiply, power]
    
toolkit =MathToolKit()
tools = toolkit.get_tools()

for t in tools:
    print(f"{t.name} => {t.description}")