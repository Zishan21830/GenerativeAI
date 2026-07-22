# The langchain.output_parsers module no longer exists in the main langchain package. Instead, many legacy components were moved into the separate langchain-classic package, while the recommended modern approach is to use structured output via Pydantic models and with_structured_output().
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    task="text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
)

model = ChatHuggingFace(llm=llm)

# Schema
schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic")
]

# Create parser
parser = StructuredOutputParser.from_response_schemas(schema)

# prompt
template  = PromptTemplate(
    template="""
    Give me 3 facts about {topic}.

    {format_instructions}
    """,
    input_variables=["topic"],
    partial_variables={
        'format_instructions' : parser.get_format_instructions()
    }
)

prompt = template.invoke({'topic': 'black hole'})
response = model.invoke(prompt)
result = parser.parse(response.content)
print(result)

# Output: {
    # 'fact_1': 'A black hole is a region in space where the gravitational pull is so strong that not even light can escape.', 
    # 'fact_2': 'Black holes are formed when a massive star collapses in on itself, causing a massive amount of matter to be compressed into a tiny space.', 
    # 'fact_3': "The point of no return around a black hole is called the event horizon, and once something crosses the event horizon, it is trapped by the black hole's gravity."
    # }