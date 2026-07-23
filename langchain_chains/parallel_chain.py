from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

# model
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)

# parser for company's summary
class CompanyAnalysis(BaseModel):
    company_name: str = Field(description="Name of the company")
    primary_products: List[str] = Field(description="What are the different primary products of the company")
    services: List[str] = Field(description="What are the different services of the company")
    industry: List[str] = Field(description="What are the different industry  where the company works")
    
parser1 = PydanticOutputParser(pydantic_object=CompanyAnalysis)

# parser for company's SWOT
class CompanySWOT(BaseModel):
    strengths: List[str] = Field(description="Write down the 2 strenths of the company")
    weaknesses: List[str] = Field(description="Write down the 2 weakness of the company")
    opportunities: List[str] = Field(description="Write down the oppertunities in the company")
    threats: List[str] = Field(description="Write down the 2 threats of the company")
    
parser2 = PydanticOutputParser(pydantic_object=CompanySWOT)

# parser for company's Use Cases
class CompanyAIUseCases(BaseModel):
    use_cases: List[str] = Field(description="Suggest practical AI applications")
    
parser3 = PydanticOutputParser(pydantic_object=CompanyAIUseCases)


# parser for company's business risks
# List the major operational, financial, technological, or market risks the company may face.
class CompanyBusinessRisk(BaseModel):
    operational: List[str] = Field(description="List the major operational risks the company may face")
    financial: List[str] = Field(description="List the major financial risks the company may face")
    technological: List[str] = Field(description="List the major technological risks the company may face")
    market: List[str] = Field(description="List the major market risks the company may face")

parser4 = PydanticOutputParser(pydantic_object=CompanyBusinessRisk)

    
"""
Problem Statement: Multi-Dimensional Company Research Assistant Using LangChain Parallel Chain
Background

You are building an AI-powered research assistant for a business analyst. The analyst frequently needs to gather different types of information about a company before making investment or partnership decisions. Instead of asking the LLM multiple questions one after another, the goal is to retrieve all the required information simultaneously to reduce response time and improve efficiency.

Objective

Implement a Parallel Chain in LangChain that processes multiple independent prompts concurrently using the same Large Language Model.

The application should accept a company name as input and generate multiple independent outputs in parallel.

Requirements

For a given company name, the system should generate the following information simultaneously:

Company Overview:
A concise summary of the company.
Include its primary products/services and industry.

SWOT Analysis:
Identify one or two Strengths, Weaknesses, Opportunities, and Threats.

Potential AI Use Cases:
Suggest practical AI applications that could improve the company's business processes.

Business Risks:
List the major operational, financial, technological, or market risks the company may face.

Expected Input:
Company Name: Tesla
Expected Output Structure:

The final result should contain the following sections:

{
    "overview": ...,
    "swot": ...,
    "ai_use_cases": ...,
    "business_risks": ...
}
"""

summary_template = PromptTemplate(
    template="""
    You are an AI-powered research assistant for a business analyst. 
    Generate a  concise summary of the company {company_name}. Include its primary products/services and industry.
    {format_instructions}
    """,
    input_variables=['company_name'],
    partial_variables={
        'format_instructions' : parser1.get_format_instructions()
    }
)
SWOT_template = PromptTemplate(
    template="""
    You are an AI-powered research assistant for a business analyst. 
    Identify one or two Strengths, Weaknesses, Opportunities, and Threats of the company {company_name}.
    {format_instructions}
    """,
    input_variables=['company_name'],
    partial_variables={
        'format_instructions' : parser2.get_format_instructions()
    }
)
use_cases_template = PromptTemplate(
    template="""
    You are an AI-powered research assistant for a business analyst. 
    Suggest practical AI applications that could improve the {company_name}'s business processes.
    {format_instructions}
    """,
    input_variables=['company_name'],
    partial_variables={
        'format_instructions' : parser3.get_format_instructions()
    }
)
business_risks_template = PromptTemplate(
    template="""
    You are an AI-powered research assistant for a business analyst. 
    List the major operational, financial, technological, or market risks the {company_name} may face.
    {format_instructions}
    """,
    input_variables=['company_name'],
    partial_variables={
        'format_instructions' : parser4.get_format_instructions()
    }
)

parallel_chains = RunnableParallel(
    {
        "summary": summary_template | model | parser1,
        "swot": SWOT_template | model | parser2,
        "use_cases": use_cases_template | model | parser3,
        "business_risks": business_risks_template | model | parser4,
    }
)

response = parallel_chains.invoke({'company_name': 'Anthropic'})
# print(response)
parallel_chains.get_graph().print_ascii()