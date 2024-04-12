from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.llms.ctransformers import CTransformers
from langchain_community.vectorstores import DeepLake
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List
from langchain_core.documents.base import Document


class DrakeLM:
    def __init__(self, model_path: str, db: DeepLake, config: dict):
        """
        Parameters:
            model_path (str): The path to the model in case running Llama
            db (DeepLake): The DeepLake DB object
            config (dict): The configuration for the llama model

        Initialize the DrakeLM model
        """
        self.gemini = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
        self.retriever = db.as_retriever()
        self.chat_history = ChatMessageHistory()
        self.chat_history.add_user_message("You are assisting a student to understand topics.")
        self.notes_prompt = load_prompt("prompt_templates/notes_prompt.yaml")
        self.chat_prompt = load_prompt("prompt_templates/chat_prompt.yaml")

    def _chat_prompt(self, query: str, context: str) -> (PromptTemplate, str):
        """
        Parameters:
            query (str): The question asked by the user
            context (str): The context retrieved from the DB

        Returns:
            PromptTemplate: The prompt template for the chat
            prompt (str): The prompt string for the chat

        Create the chat prompt for the LLM model
        """
        prompt = """You are assisting a student to understand topics. \n\n 
         You have to answer the below question by utilising the below context to answer the question. \n\n 
         Note to follow the rules given below \n\n 
         Question: {query} \n\n 
         Context: {context} \n\n
         Rules: {rules} \n\n
         Answer: 
         """

        rules = """
        - If the question says answer for X number of marks, you have to provide X number of points.
        - Each point has to be explained in 3-4 sentences.
        - In case the context express a mathematical equation, provide the equation in LaTeX format as shown in the example.
        - In case the user requests for a code snippet, provide the code snippet in the language specified in the example.
        - If the user requests to summarise or use the previous message as context ignoring the explicit context given in the message.
         """

        prompt = prompt.format(query=query, context=context, rules=rules)
        return PromptTemplate.from_template(prompt), prompt

    def _retrieve(self, query: str, metadata_filter, k=3, distance_metric="cos") -> str:
        """
        Parameters:
            query (str): The question asked by the user
            metadata_filter (dict): The metadata filter for the DB
            k (int): The number of documents to retrieve
            distance_metric (str): The distance metric for retrieval

        Returns:
            str: The context retrieved from the DB

        Retrieve the context from the DB
        """
        self.retriever.search_kwargs["distance_metric"] = distance_metric
        self.retriever.search_kwargs["k"] = k

        if metadata_filter:
            self.retriever.search_kwargs["filter"] = {
                "metadata": {
                    "id": metadata_filter["id"]
                }
            }

        retrieved_docs = self.retriever.get_relevant_documents(query)

        context = ""
        for rd in retrieved_docs:
            context += "\n" + rd.page_content

        return context

    def ask_llm(self, query: str, metadata_filter: dict = None) -> str:
        """
        Parameters:
            query (str): The question asked by the user
            metadata_filter (dict): The metadata filter for the DB

        Returns:
            str: The response from the LLM model

        Ask the LLM model a question
        """
        context = self._retrieve(query, metadata_filter)
        print("Retrieved context")
        prompt_template, prompt_string = self._chat_prompt(query, context)
        self.chat_history.add_user_message(prompt_string)
        print("Generating response...")

        rules = """
        - If the question says answer for X number of marks, you have to provide X number of points.
        - Each point has to be explained in 3-4 sentences.
        - In case the context express a mathematical equation, provide the equation in LaTeX format as shown in the example.
        - In case the user requests for a code snippet, provide the code snippet in the language specified in the example.
        - If the user requests to summarise or use the previous message as context ignoring the explicit context given in the message.
        """

        prompt_template = self.chat_prompt.format(query=query, context=context, rules=rules)
        self.chat_history.add_ai_message(AIMessage(content=self.gemini.invoke(prompt_template).content))

        return self.chat_history.messages[-1].content

    def create_notes(self, documents: List[Document]) -> str:
        """
        Parameters:
            documents (List[Document]): The list of documents to create notes from

        Returns:
            str: The notes generated from the LLM model

        Create notes from the LLM model
        """
        rules = """
        - Follow the Markdown format for creating notes as shown in the example.
        - The heading of the content should be the title of the markdown file.
        - Create subheadings for each section.
        - Use numbered bullet points for each point.   
        """

        notes_chunk = []
        for doc in documents:
            prompt = self.notes_prompt.format(content_chunk=doc.page_content, rules=rules)
            response = self.gemini.invoke(prompt)
            notes_chunk.append(response.content)

        return '\n'.join(notes_chunk)
