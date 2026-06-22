import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv
from langchain_community.chat_models import ChatLiteLLM



load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
# google_api_key = os.getenv('GEMINI_API_KEY')
# openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

MODEL = "llama-3.1-8b-instant"

# DB_NAME = str(Path(__file__).parent.parent / "vector_db")
# Force absolute resolution from root drive
CURRENT_FILE = Path(__file__).resolve() # implementation/answer.py
PROJECT_ROOT = CURRENT_FILE.parent.parent # project_5/
DB_NAME = str(PROJECT_ROOT / "vector_db")

print(f"📡 [DEBUG] answer.py loading database from: {DB_NAME}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
RETRIEVAL_K = 5

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
# Use standard similarity retrieval for testing
# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 4}
# )

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

llm = ChatGroq(temperature=0, model_name=MODEL)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.0)
# llm = ChatLiteLLM(
#     model="openrouter/openai/gpt-oss-120b", 
#     temperature=0.0
# )


# def fetch_context(question: str) -> list:
#     """
#     Retrieve relevant context documents for a question.
#     """
#     print(f"\n🔍 [RETRIEVER] Querying DB for: '{question}'")
#     docs = retriever.invoke(question)
#     print(f"🎯 [RETRIEVER] Found {len(docs)} matching document chunks.")
    
#     for i, doc in enumerate(docs):
#         src = doc.metadata.get('source', 'Unknown source')
#         print(f"   👉 Chunk {i+1} from '{os.path.basename(src)}': {doc.page_content[:80]}...")
        
#     return docs

def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


# def combined_question(question: str, history: list[dict] = []) -> str:
#     """
#     Combine all the user's messages into a single string.
#     """
#     prior = "\n".join(m["content"] for m in history if m["role"] == "user")
#     return prior + "\n" + question

def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    # EMERGENCY DEFENSE: If 'question' itself accidentally came in as a list, turn it into a string
    if isinstance(question, list):
        if len(question) > 0 and isinstance(question[0], dict):
            question = question[-1].get("content", "")
        else:
            question = " ".join(str(q) for q in question)

    # Convert the list of message dicts into raw strings safely
    user_messages = []
    if isinstance(history, list):
        for m in history:
            if isinstance(m, dict) and m.get("role") == "user":
                user_messages.append(str(m.get("content", "")))
    
    # If there are no prior messages, return just the question string
    if not user_messages:
        return str(question)
        
    prior = "\n".join(user_messages)
    return f"{prior}\n{question}"

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs