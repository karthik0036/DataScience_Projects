from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate   
from langchain_groq import ChatGroq 
from langchain_core.output_parsers import StrOutputParser 
from langserve import add_routes

import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)
llm

generic_template = "Translate the text into given {language}"

prompt = ChatPromptTemplate.from_messages(
    [
        
        ("system",generic_template),("user","{text}")
    ]
)

parser = StrOutputParser()


chain = prompt | llm | parser 

#create a app 

app = FastAPI( title="Langchain Server" ,
               version = "1.0",
               description = "A simple API server usiing Langchain runnable interfaces") 


add_routes(
    app,
    chain,
    path="/chains"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)