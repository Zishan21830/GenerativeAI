from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    task="text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
)

model = ChatHuggingFace(llm=llm)

# Without String Output parsers (StrOutputParser)
# report_detailed_template = PromptTemplate(
#     template="Write a detailed report on {topic}",
#     input_variables=["tpoic"]
# )

# report_summary_template = PromptTemplate(
#     template="Write a 5 line summary on the following text. /n {text}",
#     input_variables = ["text"]
# )

# report_detail_prompt = report_detailed_template.invoke({'topic': "black hole"})
# report_detail = model.invoke(report_detail_prompt)

# report_summary_prompt = report_summary_template.invoke({'text': report_detail.content})
# report_summary = model.invoke(report_summary_prompt)
# print(report_summary.content)

# With String Output parsers (StrOutputParser)
# Prompt to generate the detailed of the report
report_detailed_template = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["tpoic"]
)

# prompt to generate the 5-line summary of the generated report's detail
report_summary_template = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables = ["text"]
)

# Extract text content from model outputs as a string. Converts model outputs (such as AIMessage or AIMessageChunk objects) into plain text strings. It's the simplest output parser and is useful when you need string responses for downstream processing, display, or storage.
parser = StrOutputParser()

# Create a chain for entire workflow
chain = report_detailed_template | model | parser | report_summary_template | model | parser
result = chain.invoke({'topic' : 'black hole'})
print(result)