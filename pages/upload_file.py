import streamlit as st
import io
import PyPDF2
from app import disable_sidebar, initialize_models
from model import DrakeLM
from utilis import Processing

disable_sidebar()
st.title('Drake')
st.subheader('Learn without the mess of making notes!')
st.divider()

if st.button("Youtube/Video URL"):
    st.switch_page("pages/upload_url.py")

st.subheader('Upload the file')
uploaded_file = st.file_uploader(label="Choose a file", type=['pdf', 'doc'])
allow_make_notes = st.toggle('Make Complete Notes!')


if uploaded_file:
    if st.button("Upload to DB"):

        # Chunking the file
        with st.spinner('Please wait, file is chunking ...'):
            try:
                pdf_stream = io.BytesIO(uploaded_file.read())
                pdf_reader = PyPDF2.PdfReader(pdf_stream)

                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                processing, drake = initialize_models()
                documents, metadata = processing.load_pdf("hello world", text)
                st.session_state["metadata"] = metadata
                st.success("Successfully chunked the file")

            except Exception as e:
                st.error("Error in chunking")

            # Uploading to DB
            with st.spinner('Please wait, file is uploading ...'):
                try:
                    processing.upload_to_db(documents)
                except Exception as e:
                    st.error("Error in uploading")

                # Generating Notes
                if allow_make_notes:
                    with st.spinner('Please wait, notes are being generated ...'):
                        try:
                            config = {"max_new_tokens": 4096, "context_length": 8192, "temperature": 0.3}
                            notes = drake.create_notes(documents)
                            encoded_text = notes.encode('utf-8')
                            st.success("Notes generated successfully")
                            if st.download_button(
                                    label="Download data as Markdown",
                                    data=encoded_text,
                                    file_name='your_notes.md',
                                    mime='text/markdown',
                            ):
                                st.switch_page("pages/chat.py")
                        except Exception as e:
                            print(e)
                            st.error("Error in generating notes")

                else:
                    st.switch_page("pages/chat.py")
