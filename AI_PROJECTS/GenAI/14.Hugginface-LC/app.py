import validators,streamlit as st
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate 
from langchain_classic.chains.summarize import load_summarize_chain


## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Huggingface API Token and url(YT or website)to be summarized

with st.sidebar:
    hf_api_key = st.text_input("Huggingface API Token",value="",type="password")

generic_url =  st.text_input("Please provide url" , label_visibility="collapsed")

repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct"
llm_endpoint=HuggingFaceEndpoint(repo_id=repo_id,max_new_tokens=300,temperature=0.7,huggingfacehub_api_token=hf_api_key)

chat_model = ChatHuggingFace(llm=llm_endpoint)

# 2. Refined Summary Prompt
prompt_template = """
<s>[INST] Write a concise summary of the following content in about 300 words. 
Return the summary in bullet points.

Content: {text} [/INST]</s>
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


if st.button("Summarize the Content from YT or Website"):
    # validations
    
    if not hf_api_key.strip() or not generic_url.strip():
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
                
                chain = load_summarize_chain(chat_model,chain_type="stuff",prompt=prompt)
                
                output_summary = chain.invoke(docs)
                
                st.success(output_summary['output_text'])
        except Exception as e:
            st.exception(f"Exception:{e}")

            
            
    

