from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import certifi

# ✅ Fix SSL issues by forcing Python to use certifi’s CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in environment.")

os.environ["GOOGLE_API_KEY"] = "AIzaSyDpOPNrDHfWHTk038JHsdeaYtL2130jJQw"
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.5)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    'tags' : ['llm app', 'report generation' , 'summarization'] ,'metadata' : {'model1' : 'gemini-1.5-flash'}
}
result = chain.invoke({'topic': 'Unemployment in India'})

print(result)
