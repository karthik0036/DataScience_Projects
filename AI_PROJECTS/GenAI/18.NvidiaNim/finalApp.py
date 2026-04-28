import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

def vector_embeddings():
    
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectorsDb = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)
        

        
st.title("Nvidia NIM Demo RAG APPLICATION")

if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("Vector Store DB Is Ready")
    
    
llm = ChatNVIDIA(model="nvidia/nemotron-3-nano-30b-a3b")


prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

user_query = st.text_input("Enter Your Question From Documents")

if user_query:
    doc_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectorsDb.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,doc_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input':user_query})
    print("Response time :",time.process_time() - start)
    st.write(response['answer'])
    
     # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
    



