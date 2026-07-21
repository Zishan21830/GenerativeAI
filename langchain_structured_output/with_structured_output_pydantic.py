"""
Models can be requested to provide their response in a format matching a given schema. This is useful for ensuring the output can be easily parsed and used in subsequent processing. LangChain supports multiple schema types and methods for enforcing structured output.
1. TypeDict
2. Pydantic
3. JSON Schema
"""

from langchain_openai import ChatOpenAI
from typing import Optional
from pydantic import BaseModel, Field, EmailStr
import datetime
from dotenv import load_dotenv

load_dotenv()

# Create model
model = ChatOpenAI()

# Pydantic
# Pydantic models provide the richest feature set with field validation, descriptions, and nested structures.

# Problem:
# "John Doe needs to book a business trip from New York to London on October 15, 2026. He prefers to fly in Business class to ensure he can rest during the long flight. Please extract his full name, the origin city, his final destination, the exact date of his travel, and his preferred seating arrangement from this request."


# a) Basic Example
# class FlightBooking(BaseModel):
#     passenger_name: str
#     source: str
#     destination: str
#     departure_date: datetime.date
#     seat_class: str
    
# b) Default values
# class FlightBooking(BaseModel):
#     passenger_name: str = "John Doe"
#     source: str = "New Delhi"
#     destination: str = "Lucknow"
#     departure_date: datetime.date = datetime.date(2026, 10, 15)
#     seat_class: str = "Economy"

# c) Optional Fields
# class FlightBooking(BaseModel):
#     passenger_name: str
#     source: str
#     destination: str
#     departure_date: datetime.date
#     seat_class: Optional[str] = None
#     email: Optional[EmailStr] = None

# d) Field Function
class FlightBooking(BaseModel):
    passenger_name: str = Field(description="Name of the passenger", default="John Doe")
    source: str = Field(description="Name of the source city of the passenger", default="New Delhi")
    destination: str = Field(description="Name of the destination city of the passenger", default = "Lucknow")
    departure_date: datetime.date
    flight_hours: Optional[str] =  Field(description="Number of hours to complete the journey. It could be in minutes and hours", gt=0)
    seat_class: Optional[str] = None
    email: Optional[EmailStr] = None

model_with_structure = model.with_structured_output(FlightBooking)
response = model_with_structure.invoke("John Doe needs to book a business trip from New York to London on October 15, 2026. He prefers to fly in Business class to ensure he can rest during the long flight. Please extract his full name, the origin city, his final destination, the exact date of his travel, and his preferred seating arrangement from this request. It will takes 1hrs and 20 minutes to complete the journey.")

print(response)   