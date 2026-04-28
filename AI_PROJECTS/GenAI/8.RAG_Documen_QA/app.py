import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings



import os 
from dotenv import load_dotenv
load_dotenv() 

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
    """ 
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question:{input}
    """
)
def create_vector_embeddings():
    if "db" not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader("research_papers") 
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.db = FAISS.from_documents(st.session_state.final_docs,st.session_state.embedding)
st.title("RAG Document Q&A with GROQ")
        
        
user_prompt = st.text_input("Enter your queries here...")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector db is ready")
    
  
import time 
  
if user_prompt: 
    if "db" not in st.session_state:
        st.warning("Please click 'Document Embedding' first")
    else:
        document_chain = create_stuff_documents_chain(llm,prompt)
        retriver = st.session_state.db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriver,document_chain)
    
        start = time.process_time()
        response = retrieval_chain.invoke({"input":user_prompt})
        print(f"time taken is :{time.process_time()-start}")
    
        st.write(response['answer'])
    
        with st.expander("Document similarity search"):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)

# Add this to see exactly what the retriever found
# st.write("Number of relevant chunks found:", len(response['context']))
    
    
    

        
    









