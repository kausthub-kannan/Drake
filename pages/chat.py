import streamlit as st
from app import disable_sidebar, initialize_models

disable_sidebar()
col1, col2= st.columns([3, 1.6])

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_notes" not in st.session_state:
    st.session_state.chat_notes = f""""""
    st.session_state.encoded_text = st.session_state.chat_notes.encode('utf-8')

col1.title('Chat with Drake!')
if col2.button("Home"):
    st.switch_page("app.py")

universal_chat = st.toggle("Universal Chat")
st.caption("Note: Universal Chat uses the complete DB to retrieve context, use it with caution")

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Drake your questions"):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Drake is thinking..."):
        query = f"{prompt}"
        _, drake = initialize_models() # Check resources in cache
        if universal_chat:
            response = drake.ask_llm(query)
        else:
            response = drake.ask_llm(query, metadata_filter=st.session_state["metadata"])

        with st.chat_message("assistant"):
            st.session_state.chat_notes += query + "\n" + response + "\n\n"
            st.session_state.encoded_text = st.session_state.chat_notes.encode('utf-8')

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.download_button(
    label="Export",
    data=st.session_state.encoded_text,
    file_name='chat_history.md',
    mime='text/markdown',
)

