import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()

#LANGSMITH TRACKING
os.environ["LANGCHAIN_API_KEY"] == os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot" 

#creating prompt 

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful massistant . Please  repsonse to the user queries"),
        ("user","Question:{question}")
        
    ]
)

def generate_response(question,engine,temperature,max_tokens):
    llm = ChatOllama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question}) 
    return answer

    

##TITLE OF THE APP
st.title("Q&A Chatbot")

## Select the OpenAI model
engine=st.sidebar.selectbox("Select Open AI model",["gemma:2b","phi3"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## MAin interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")


