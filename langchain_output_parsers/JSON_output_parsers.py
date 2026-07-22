from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    task="text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
)

model = ChatHuggingFace(llm=llm)

# Create parser
parser = JsonOutputParser()

# prompt
template  = PromptTemplate(
    template="Give me the name, age and city of the scifi person \n {formate_instruction}",
    input_variables=[],  #We don't have any input variable.
    partial_variables={
        'formate_instruction' : parser.get_format_instructions()
    }
)
# prompt = template.format()
# result = model.invoke(prompt)
# finale_result = parser.parse(result.content)
# print(finale_result)

# Using chain
chain = template | model | parser
result = chain.invoke({})  # pass empty dect as input becuase we don't have any input variable.
print(result)
# Output: {'name': 'Neo', 'age': 35, 'city': 'New York City'}

# The draw back of JsonOutputParser is we can't enforce the JSON schema.