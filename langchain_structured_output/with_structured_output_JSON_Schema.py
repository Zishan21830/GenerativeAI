"""
Models can be requested to provide their response in a format matching a given schema. This is useful for ensuring the output can be easily parsed and used in subsequent processing. LangChain supports multiple schema types and methods for enforcing structured output.
1. TypeDict
2. Pydantic
3. JSON Schema
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os


# Create model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    
) # type: ignore
model = ChatHuggingFace(llm=llm)

# JSON Schema
json_schema = {
  "title": "EventOrganizer",
  "type": "object",
  "description": "Details of a person who is attending a specific event.",
  "properties": {
    "event_name": {
      "type": "string",
      "description": "The official name or title of the event."
    },
    "participant_name": {
      "type": "string",
      "description": "The full name of the person attending the event."
    },
    "email": {
      "type": "string",
      "description": "The contact email address of the participant."
    },
    "phone": {
      "type": "integer",
      "description": "The participant's phone number, including area code, containing only digits."
    },
    "ticket_type": {
      "type": "string",
      "description": "The category of ticket purchased, such as General Admission, VIP, or Early Bird."
    }
  },
  "required": [
    "event_name",
    "participant_name",
    "phone",
    "ticket_type"
  ]
}



model_with_structure = model.with_structured_output(json_schema)
response = model_with_structure.invoke("Hey Team, just wanted to log a new registration that came in over the phone. Sarah Jenkins called to confirm her spot for the upcoming Tech Pioneers Summit 2026. She bought a VIP pass to get access to the workshops. Her phone number is 5550192834. She mentioned she hasn't checked her email yet today, but her address on file is sarah.j@techmail.com.")

print(response)   
# Output: {'event_name': 'Tech Pioneers Summit 2026', 'participant_name': 'Sarah Jenkins', 'email': 'sarah.j@techmail.com', 'phone': 5550192834, 'ticket_type': 'VIP'}