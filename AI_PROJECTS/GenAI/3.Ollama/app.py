import os
from dotenv import load_dotenv 


import streamlit as st
# from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM



load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

st.title("LLM WITH OLLAMA-GEMMA:2B")
input_text = st.text_input("whats on yoour mind?")

prompt = ChatPromptTemplate.from_messages(
    [
    ("system","you are helpful assistant , please respond to the questions"),
    ("user","{question}")
    
    ]
)


llm = OllamaLLM(model="gemma:2b")

parser = StrOutputParser()

chain = prompt | llm | parser 

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)