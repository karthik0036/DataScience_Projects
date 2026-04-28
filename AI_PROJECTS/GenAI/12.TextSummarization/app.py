import validators,streamlit as st
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
from langchain_classic.chains.summarize import load_summarize_chain


## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and url(YT or website)to be summarized

with st.sidebar:
    groq_api_key = st.text_input("Please provide API key",value="",type="password")

generic_url =  st.text_input("Please provide url" , label_visibility="collapsed")

llm = ChatGroq(model_name="llama-3.1-8b-instant",groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
    """
    Provide a summary of the following content in 300 words:
    Content:{text}
    """
)


if st.button("Summarize the Content from YT or Website"):
    # validations
    
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the API key and the URL.")
    elif not validators.url(generic_url): 
        st.error("The URL provided is not valid. Please check and try again.")
    else:
        try:
            with st.spinner("Waiting..."):
                
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url, 
                        add_video_info=True,
                        language=["en", "en-US", "a.en"], # Good for multi-lingual videos
                        translation="en",
                        continue_on_failure=True
                    )
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()
                
                chain = load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                
                output_summary = chain.invoke(docs)
                
                st.success(output_summary['output_text'])
        except Exception as e:
            st.exception(f"Exception:{e}")

            
            
    

