import streamlit as st
from langchain_groq import ChatGroq
# LLMMathChain is legacy; modern math apps use custom tools or LLM tool-calling
from langchain_community.tools.tavily_search import TavilySearchResults # Better than Wikipedia in 2026
from langchain.agents import create_agent
# from langchain.agents import create_react_agent, AgentExecutor
# In 1.x, AgentExecutor is often re-exported, but if it fails:
from langchain_classic.agents import AgentExecutor,create_react_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.tools import tool # The new way to define math tools
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.output_parsers import StrOutputParser
import numexpr



## Set up the Stramlit app
st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver")

groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()
    
llm=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)

##TOOLS INIT

wikipedia = WikipediaAPIWrapper()
@tool
def wikipedia_tool(query: str) -> str:
    """
    Search Wikipedia for various information on topics, historical events, 
    or general knowledge. Input should be a search query.
    """
    return wikipedia.run(query)


@tool
def math_tool(expression: str) -> str:
    """Useful for calculating mathematical expressions. Input: '2+2' or 'exp(3)'."""
    try:
        # numexpr.evaluate handles the parsing and safety for you
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"Math Error: {e}"
    
prompt = ChatPromptTemplate.from_template(
    """
        Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
        and display it point wise for the question below
        Question:{question}
        Answer:
    """
)

# This is a simple 'LCEL' chain
reasoning_chain = prompt | llm | StrOutputParser()

# 2. Define the tool using the @tool decorator
@tool
def reasoning_tool(query: str) -> str:
    """
    Useful for answering logic-based, step-by-step reasoning, 
    and complex brain-teaser questions. Input should be a 
    detailed description of the problem.
    """
    # .invoke() is the 2026 standard replacement for .run()
    return reasoning_chain.invoke({"question": query})

tools = [wikipedia_tool, math_tool, reasoning_tool]

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI agent.

You have access to the following tools:
{tools}

Use the following format:

Question: {input}
Thought: think step by step
Action: one of [{tool_names}]
Action Input: input to the tool
Observation: result of the tool
... (repeat Thought/Action if needed)
Final Answer: final answer to the user

Begin!

Question: {input}
{agent_scratchpad}
""")

# ✅ Create agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ✅ Executor (this replaces initialize_agent runtime)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a MAth chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## LEts start the interaction
question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("find my answer"):
    if question:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Generating response..."):
            # 1. Setup the Callback
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            
            # 2. Use .invoke() instead of .run()
            # We pass 'question' to the 'input' key defined in your ChatPromptTemplate
            response = agent_executor.invoke(
                {"input": question}, 
                {"callbacks": [st_cb]}
            )
            
            # 3. Extract the answer from the response dictionary
            answer = response["output"]
            
            # 4. Save to history and display
            st.session_state.messages.append({'role': 'assistant', "content": answer})
            st.write('### Response:')
            st.success(answer)
    else:
        st.warning("Please enter a question")