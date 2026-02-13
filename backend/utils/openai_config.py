from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

def get_openai_llm():
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in .env")

    os.environ["OPENAI_API_KEY"] = api_key
    return ChatOpenAI(model=model)