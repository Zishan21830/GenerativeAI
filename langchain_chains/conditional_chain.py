from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",  # Use a valid model name
    temperature=0
)

# Output Schema
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )

parser = PydanticOutputParser(pydantic_object=Feedback)

# Sentiment Classification Prompt
prompt = PromptTemplate(
    template="""
Classify the sentiment of the following feedback as either
'positive' or 'negative'.

Feedback:
{feedback}

{format_instructions}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# Response Prompts
prompt_pos_feedback = PromptTemplate(
    template="""
Write an appropriate response to this positive feedback.

Feedback:
{feedback}
""",
    input_variables=["feedback"],
)

prompt_neg_feedback = PromptTemplate(
    template="""
Write an appropriate response to this negative feedback.

Feedback:
{feedback}
""",
    input_variables=["feedback"],
)

# Classifier
classifier_chain = prompt | model | parser
# Preserve original feedback while adding sentiment
classifier = RunnablePassthrough.assign(
    sentiment=classifier_chain
)

# Branch
branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].sentiment == "positive",
        prompt_pos_feedback | model,
    ),
    (
        lambda x: x["sentiment"].sentiment == "negative",
        prompt_neg_feedback | model,
    ),
    RunnableLambda(lambda _: "Couldn't determine the sentiment."),
)

# Final Chain
chain = classifier | branch_chain

response = chain.invoke(
    {
        "feedback": "It is the terrible smartphone."
    }
)

print(response.content)