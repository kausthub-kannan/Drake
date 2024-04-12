import os
import json
from typing import List
import assemblyai as aai
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.vectorstores import DeepLake
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from typing import Dict
import uuid
from pdf2image import convert_from_bytes
import pytesseract
from pytesseract import Output


class Processing:
    def __init__(self, dataset_path: str, embedding_model_name: str, chunk_size=500, chunk_overlap=5):
        """
        Parameters:
            dataset_path (str): Path to the dataset in the Vector-DB
            embedding_model_name (str): Name of the HuggingFace model to be used for embeddings
            chunk_size (int): Size of each chunk to be processed
            chunk_overlap (int): Overlap between each chunk

        Initialize embedding model, text splitter, transcriber and Vector-DB
        """
        self.dataset_path = dataset_path
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.transcriber = aai.Transcriber()

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={'normalize_embeddings': False}
        )

        self.db = DeepLake(dataset_path=f"hub://{self.dataset_path}",
                           embedding=self.embedding_model,
                           exec_option="compute_engine"
                           )

    def _add_metadata(self, documents: List[Document], url: str, id: str, source: str, file_type: str,
                      course_tag="") -> (List[Document], Dict[str, str]):
        """
        Parameters:
            documents (List[Document]): List of documents to add metadata to
            id (str): ID of the documents
            source (str): Source of the documents
            file_type (str): Type of the documents
            course_tag (str): Tag to identify the course the documents belongs to

        Returns:
            documents (List[Document], Dict[str, str): List of documents with metadata added along with the metadata

        Add metadata to the documents
        """
        metadata = {
            "id": id,
            "source": source,
            "url": url,
            "file_type": file_type,
            "course_tag": course_tag
        }
        for doc in documents:
            doc.metadata = metadata
        return documents, metadata

    def load_pdf(self, text) -> (List[Document], Dict[str, str]):
        """
        Returns:
            documents (List[Document], Dict[str, str): List of documents with metadata added along with the metadata

        Load PDF file, split into chunks and add metadata
        """
        pdf_chunk = self.text_splitter.create_documents([text])
        print("Created document chunks")
        return self._add_metadata(pdf_chunk, url="NaN", id=str(uuid.uuid4()), source="document", file_type="pdf")

    def load_scanned_pdf(self, file) -> str:
        """
        Parameters:
            file (File): Scanned PDF file to be processed

        Returns:
            str: Text extracted from the scanned PDF file

        Extract text from scanned PDF file
        """
        images = convert_from_bytes(file)

        all_text = ""
        for image in images:
            # Perform OCR on the image
            text = pytesseract.image_to_data(image, lang='eng', output_type=Output.DICT)

            # Extract text from the dictionary
            page_text = " ".join(text['text'])
            all_text += page_text + "\n"

        return all_text

    def load_transcript(self, url) -> (List[Document], Dict[str, str]):
        """
        Returns:
            documents (List[Document], Dict[str, str): List of documents with metadata added along with the metadata

        Load transcript, split into chunks and add metadata
        """
        transcript = self.transcriber.transcribe(url)
        print("Transcribed")
        transcript_chunk = self.text_splitter.create_documents([transcript.text])
        print("Created transcript chunks")
        return self._add_metadata(transcript_chunk, url="NaN", id=str(uuid.uuid4()), source="custom_video",
                                  file_type="transcript")

    def load_yt_transcript(self, url) -> (List[Document], Dict[str, str]):
        """
        Returns:
            documents (List[Document], Dict[str, str): List of documents with metadata added along with the metadata

        Load YouTube transcript, split into chunks and add metadata
        """
        if url.startswith("https://www.youtube.com/watch?v="):
            video_id = url.replace("https://www.youtube.com/watch?v=", "")
        else:
            video_id = url.replace("https://youtu.be/", "")

        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        print("Downloaded transcript")
        transcript = [line['text'] for line in transcript]
        transcript_text = ' '.join(transcript)
        yt_transcript_chunk = self.text_splitter.create_documents([transcript_text])
        print("Created YouTube transcript chunks")
        return self._add_metadata(yt_transcript_chunk, url=url, id=video_id, source="youtube", file_type="transcript")

    def upload_to_db(self, documents: List[Document]):
        """
        Parameters:
            documents (List[Document]): List of documents to upload to Vector-DB

        Upload documents to Vector-DB
        """
        print("Embedding and uploading to Vector-DB...")
        self.db.add_documents(documents)
        print("Uploaded to Vector-DB")


class PromptCreate:
    def __init__(self, example_path, save_path):
        self.examples = []
        self.example_prompt = None
        self.few_shot_prompt = None
        self.example_path = example_path
        self.file_name = "example_{i}.json"
        self.save_path = save_path

    def load_examples(self):
        for i in range(1, len(os.listdir(self.example_path)) + 1):
            filename = os.path.join(self.example_path, self.file_name.format(i=i))
            try:
                with open(filename, "r") as json_file:
                    self.examples.append(json.load(json_file))
            except FileNotFoundError:
                print(f"File {filename} not found.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file {filename}.")

    def create_prompt_template(self, input_variables, template_string):
        self.example_prompt = PromptTemplate(input_variables=input_variables, template=template_string)

    def create_few_shot_prompt(self, prefix, suffix):
        self.few_shot_prompt = FewShotPromptTemplate(
            examples=self.examples, example_prompt=self.example_prompt, prefix=prefix, suffix=suffix
        )

    def save_prompt(self):
        self.few_shot_prompt.save(self.save_path)
