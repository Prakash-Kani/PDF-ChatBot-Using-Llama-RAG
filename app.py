import streamlit as st
import os
import glob
import pandas as pd

from chatbot import Conversational_Chain



st.set_page_config(page_title = "Custom PDF ChatBot",
                   page_icon = "https://cdn.emojidex.com/emoji/seal/youtube.png",
                   layout = "wide",
                   initial_sidebar_state = "expanded",
                   menu_items = None)

dic = {'user_name':[], 'SessionID':[], 'Prompt':[], 'LLM_Response':[]}
df = pd.DataFrame(dic)

st.cache_resource()
def model():
    return Conversational_Chain()
llm = model()





st.title(":blue[Custom PDF ChatBot Using Llama LLM (RAG)]")#📡

tab1, tab2 = st.tabs(['PDF Ingest', 'Chat Bot'])

with tab1:
    col1, col2 = st.columns([1,1], gap = 'medium')

    with col1:
        file_upload = col1.file_uploader(
            "Upload a PDF file ↓", type="pdf", accept_multiple_files=True
        )
        if file_upload is not None:
            # file_path = file_upload.names
            st.write([file_upload[i].name for i in range(len(file_upload))])
            # st.write(find_file_path(file_upload[0].name))
            

            # llm = llm_model(file_upload.name)
        else:
            st.warning("Please upload a PDF file first.")


with tab2:
    user_name = st.text_input('Enter the User Name')
    session_id = st.text_input('Enter the Session ID')


    message_container = st.container(height=500, border=True)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content":"Let's dive into chemical bonding and molecular structure. What specific topic or question can I help you with today?"}]
    # st.session_state.messages.append(
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with message_container.chat_message(message["role"]):
            st.markdown(message["content"])
        # with message_container.chat_message('assistant'):
    
    # React to user input
    if prompt := st.chat_input("Enter Your Prompt here..."):
        # Display user message in chat message container
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        message_container.chat_message("user").markdown(prompt)

        response = f"User: {prompt}"
        # Display assistant response in chat message container
        with message_container.chat_message("assistant"):
                    with st.spinner(":blue[processing...]"):
                        if llm and session_id: 
                            response = llm.invoke(
                                                    {"input": prompt},
                                                    config={"configurable": {"session_id": session_id}},
                                                    )["answer"]

                            st.markdown(response)
                            # st.markdown(docs)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            df.loc[len(df.index)] = [user_name, session_id, prompt, response]
                        else:
                            st.warning("Please upload a PDF file first.")
    st.table(df)



