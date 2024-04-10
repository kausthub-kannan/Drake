import streamlit as st
from app import disable_sidebar, initialize_models
from model import DrakeLM
from utilis import Processing

disable_sidebar()
col1, col2 = st.columns([1.2, 0.3])

col1.title('Chat with Drake!')
if col2.button("Home"):
    st.switch_page("app.py")

universal_chat = st.toggle("Universal Chat")
st.caption("Note: Universal Chat uses the complete DB to retrieve context, use it with caution")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Drake your questions"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Drake is thinking..."):
        query = f"{prompt}"
        _, drake = initialize_models()
        if universal_chat:
            response = drake.ask_llm(query)
        else:
            response = drake.ask_llm(query, metadata_filter=st.session_state["metadata"])

        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
