import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
# from langchain.agents import initialize_agent,AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import tool



## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
@tool
def arxiv(query: str) -> str:
    """Useful for searching research papers and technical topics."""
    return arxiv_wrapper.run(query)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
# wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

@tool
def wikipedia(query: str) -> str:
    """Useful for general knowledge,definitions and explanations."""
    return api_wrapper.run(query)

search_tool=DuckDuckGoSearchRun(name="Search")

@tool
def search(query: str) -> str:
    """Useful for current events general web search."""
    return search_tool.run(query)


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    
    
    llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant",streaming=True)
    tools=[search,arxiv,wikipedia]
    
    search_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt = """
            You are a helpful assistant.

            You can use tools like:
            - Wikipedia
            - Arxiv
            - Web search

            Rules:
            - Use tools only when needed.
            - Do NOT call the same tool repeatedly.
            """
            )
    
    with st.chat_message("assistant"):
        # st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.invoke(
            {
                "messages": st.session_state.messages[-5:]
            },config={"recursion_limit": 10}
            
        )

        answer = response["messages"][-1].content

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.write(answer)
        


# In older versions, the agent was a "black box." Now, with create_react_agent, you are explicitly using the ReAct logic:

# Thought: The AI decides what it needs to do.

# Action: It selects a tool (like Wikipedia).

# Action Input: It decides what to search for.

# Observation: It reads the result.

# Final Answer: It combines everything for you.
