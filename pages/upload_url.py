import streamlit as st
from app import disable_sidebar, initialize_models

# Upload Template
disable_sidebar("Drake | Upload URL")
processing, drake = initialize_models()
st.title('Drake')
st.subheader('Learn without the mess of making notes!')
st.divider()

if st.button("PDF/Transcript"):
    st.switch_page("pages/upload_file.py")

st.subheader('Enter the Video URL')
video_url = st.text_input(label="Enter the URL")
st.caption("Note: Currently, Drake support Gemini, Llama support to be added soon!")

allow_make_notes = st.toggle('Make Complete Notes!')


if video_url:
    # Upload to DB
    if st.button("Upload to DB"):

        # Chunking the file
        with st.spinner('Please wait, file is chunking ...'):
            try:
                documents, metadata = processing.load_yt_transcript(video_url)
                st.session_state["metadata"] = {"id": metadata["id"]}
            except Exception as e:
                print(e)
                st.error("Error in chunking")

            # Uploading to DB
        with st.spinner('Please wait, documents uploading ...'):
            try:
                processing.upload_to_db(documents)
                st.success("Successfully uploaded the file")
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
                        label="Download your notes",
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


