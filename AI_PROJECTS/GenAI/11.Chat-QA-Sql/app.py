import streamlit as st
from pathlib import Path
# --- THE OLD (DEPRECATED) ---
# from langchain.sql_database import SQLDatabase
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents.agent_types import AgentType
# --- THE NEW (2026 STANDARD) ---
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq # Or your preferred LLM
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

##page title
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

#interact with 2db's
LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"

#sidebar options
radio_options = ["Use SQLLite 3 Database- Student.db","Connect to  MySQL Database"]
selected_opt=st.sidebar.radio(label="Choose the DB which you want to chat",options=radio_options)

if radio_options.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide Mysql host")
    mysql_user = st.sidebar.text_input("Mysql User")
    mysql_password = st.sidebar.text_input("Mysql password",type="password")
    mysql_db = st.sidebar.text_input("Mysql Db Name")
else:
    db_uri=LOCALDB
    
api_key = st.sidebar.text_input(label="GROQ API KEY",type="password")
    
if not db_uri:
    st.info("Please provide db info")
if not api_key:
    st.info("Please provide api key")
    
## LLM model
llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant",streaming=True)

# CONNECTING THE DATABASE
@st.cache_resource(ttl="2h")
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_uri==LOCALDB:
        dbfilepath=(Path(__file__).parent/"student.db").absolute()
        print(dbfilepath)
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri==MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}")) 

if db_uri==MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db=configure_db(db_uri)
    
## toolkit
toolkit=SQLDatabaseToolkit(db=db,llm=llm)

# Fix: We use a string for agent_type instead of the old AgentType enum
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="tool-calling", # Standard for 2026; handles SQL logic perfectly
    handle_parsing_errors=True
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query=st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container())
        # CRITICAL FIX: Use .invoke() instead of .run()
        # In 2026, .run() will throw an AttributeError
        try:
            response = agent.invoke(
                {"input": user_query},
                {"callbacks": [st_cb]}
            )
            
            final_answer = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.write(final_answer)
            
        except Exception as e:
            st.error(f"Error: {e}")
            
            
# app is:

# 👉 A LangChain SQL Agent-based chatbot
# 👉 Uses Groq LLM + Streamlit UI
# 👉 Converts natural language → SQL → results
# 👉 Supports SQLite + MySQL
# 👉 Shows agent reasoning via callbacks