import streamlit as st
import os
from dotenv import load_dotenv

# Core LangChain Imports
from langchain_groq import ChatGroq
# from langchain_community.agents import AgentExecutor, create_react_agent
# from langchain import hub
from langchain_core.tools import tool
from langchain.agents import create_agent
# Tools and Utilities
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

load_dotenv()

# --- 1. TOOL DEFINITIONS ---
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
@tool
def arxiv(query: str) -> str:
    """Useful for searching research papers and technical topics."""
    return arxiv_wrapper.run(query)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
@tool
def wikipedia(query: str) -> str:
    """Useful for general knowledge, definitions, and explanations."""
    return api_wrapper.run(query)

search_tool = DuckDuckGoSearchRun()
@tool
def search(query: str) -> str:
    """Useful for current events and general web searches."""
    return search_tool.run(query)

tools = [search, arxiv, wikipedia]

# --- 2. STREAMLIT UI SETUP ---
st.set_page_config(page_title="LangChain - Chat with Search", page_icon="🔎")
st.title("🔎 LangChain - Chat with Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display existing chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# # --- 1. THE THREADING FIXER ---
# def get_streamlit_cb(container):
#     """Bridge the gap between LangChain threads and Streamlit UI."""
#     cb = StreamlitCallbackHandler(container)
#     # Capture the context of the current browser session
#     ctx = get_script_run_ctx()
    
#     # This wrapper forces the background thread to recognize the Streamlit session
#     def with_streamlit_context(func):
#         def wrapper(*args, **kwargs):
#             add_script_run_ctx(ctx)
#             return func(*args, **kwargs)
#         return wrapper

#     # Apply the context fix to the specific events that were crashing
#     cb.on_tool_start = with_streamlit_context(cb.on_tool_start)
#     cb.on_tool_end = with_streamlit_context(cb.on_tool_end)
#     cb.on_llm_new_token = with_streamlit_context(cb.on_llm_new_token)
    
#     return cb
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# --- 3. AGENT LOGIC ---
if prompt := st.chat_input(placeholder="What is the latest news on SpaceX?"):
    # Store and display user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.error("Please add your Groq API key in the sidebar to continue.")
        st.stop()

    # Initialize LLM & Agent Components
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", streaming=True)

    # create_agent is the NEW standard. It replaces create_react_agent + AgentExecutor.
    # It returns a 'Runnable' that handles the thinking loop automatically.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a professional researcher. Use tools to provide factual answers."
    )

    # Generate Response
    with st.chat_message("assistant"):
        # 1. Create the standard handler
        st_cb = StreamlitCallbackHandler(st.container())
        
        try:
            # 2. THE FIX: Use 'config' to disable parallel execution.
            # This forces the agent to stay on the same thread as Streamlit.
            response = agent.invoke(
                {"input": prompt},
                config={
                    "callbacks": [st_cb],
                    "max_concurrency": 1  # <--- CRITICAL FIX
                }
            )
            
            st.write(response["output"])
            
        except Exception as e:
            # Fallback: If the UI still hangs, just run it without the bubbles
            st.warning("Switching to standard mode...")
            response = agent.invoke({"input": prompt})
            st.write(response["output"])
            
  
#   == ERROR ===          
# 2026-04-02 08:48:25.654 Thread 'ThreadPoolExecutor-5_0': missing ScriptRunContext! This warning can be ignored when running in bare mode
# 2026-04-02 08:48:27.435 Thread 'ThreadPoolExecutor-6_0': missing ScriptRunContext! This warning can be ignored when running in bare mode
# 2026-04-02 08:48:27.435 Thread 'ThreadPoolExecutor-6_0': missing ScriptRunContext! This warning can be ignored when running in bare mode
# 2026-04-02 08:48:27.436 Thread 'ThreadPoolExecutor-6_0': missing ScriptRunContext! This warning can be ignored when running in bare mode
# Error in StreamlitCallbackHandler.on_tool_end callback: NoSessionContext()           
# Why this happens in 2026
# Older versions of LangChain ran everything "linearly" in 
# a single thread, so Streamlit never lost track of where it was. 
# The 2026 Standard (create_agent) is built on a graph architecture that uses asynchronous 
# and multi-threaded workers to make tools (like your Search and Arxiv) run much faster.


# 2026-04-02 08:54:18.614 Thread 'ThreadPoolExecutor-3_0': missing ScriptRunContext! This warning can be ignored when running in bare mode.
# 2026-04-02 08:54:18.619 Thread 'ThreadPoolExecutor-3_0': missing ScriptRunContext! This warning can be ignored when running in bare mode.
# Error in StreamlitCallbackHandler.on_tool_start callback: NoSessionContext()
# 2026-04-02 08:54:18.643 Thread 'ThreadPoolExecutor-4_0': missing ScriptRunContext! This warning can be ignored when running in bare mode.
# 2026-04-02 08:54:18.645 Thread 'ThreadPoolExecutor-4_0': missing ScriptRunContext! This warning can be ignored when running in bare mode.
# Error in StreamlitCallbackHandler.on_tool_start callback: NoSessionContext()

# The NoSessionContext() error is a classic "threading" conflict. Streamlit is 
# designed to run in a single thread per user session. However, the modern LangChain 1.0 (create_agent) 
# architecture uses background threads (ThreadPoolExecutor) to run your tools (Wikipedia, Arxiv, Search) 
# in parallel to save time.

# When the agent starts a background thread to run a tool, that 
# thread doesn't "know" which Streamlit browser tab it belongs to, 
# so the StreamlitCallbackHandler crashes when it tries to draw a UI element.

# The Solution: Manual Context Injection
# To fix this, we have to "teleport" the Streamlit context from 
# the main thread into the background threads using add_script_run_ctx            
            
            