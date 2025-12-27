from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import certifi

# ✅ Fix SSL issues by forcing Python to use certifi’s CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()

# Load environment variables from .env
load_dotenv(dotenv_path=".env")


api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in environment.")

os.environ["GOOGLE_API_KEY"] = "AIzaSyDpOPNrDHfWHTk038JHsdeaYtL2130jJQw"
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")


parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)
