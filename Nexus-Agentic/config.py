from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

# OpenAI Configuration
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("Missing required OpenAI API key. Please check your .env file.")

# Initialize OpenAI
llm = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4",
    temperature=0.7
)