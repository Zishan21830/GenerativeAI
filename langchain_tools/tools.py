# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.tools import ShellTool
from langchain_core.tools import tool

### Built-in Tools



# search_tool = DuckDuckGoSearchRun()
# results = search_tool.invoke('Iowa State University')

# shell_tool = ShellTool()
# result = shell_tool.invoke('whoami')


### Custom Tools in Langchain"""

@tool
def multiply(a: int, b: int) -> int:
  """Multiply two numbers"""
  return a*b

result = multiply.invoke({'a': 3, 'b': 5})
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

# Output:
# multiply
# Multiply two numbers
# {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}

# print(multiply.args_schema.model_json_schema())
# Output: 
# {'description': 'Multiply two numbers', 'properties': {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}, 'required': ['a', 'b'], 'title': 'multiply', 'type': 'object'}