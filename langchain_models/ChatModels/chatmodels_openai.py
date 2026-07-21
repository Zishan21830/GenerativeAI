from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4", temperature=0.2, max_completion_token=40)
result = model.invoke("What is the capital of India")
print(result.content)