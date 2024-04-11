import streamlit as st
from model import DrakeLM
from utilis import Processing

initial_page = "pages/upload_url.py"


@st.cache_resource()
def initialize_models():
    processing = Processing(
        dataset_path=st.secrets["DEEPLAKE_DB_URL"],
        embedding_model_name="mixedbread-ai/mxbai-embed-large-v1",
        chunk_size=1300,
    )
    config = {"max_new_tokens": 4096, "context_length": 8192, "temperature": 0.3}
    drake = DrakeLM(
        model_path="Mistral/mistral-7b-instruct-v0.2.Q5_K_S.gguf",
        config=config,
        db=processing.db,
    )

    return processing, drake


def disable_sidebar(page_title: str):
    st.set_page_config(page_title=page_title,
                       page_icon=None,
                       layout="centered",
                       initial_sidebar_state="collapsed",
                       )
    no_sidebar_style = """
        <style>
            div[data-testid="stSidebarNav"] {display: none;}
        </style>
    """
    st.markdown(no_sidebar_style, unsafe_allow_html=True)


def main():
    disable_sidebar("Drake")
    initialize_models()
    st.switch_page(initial_page)


if __name__ == "__main__":
    main()
