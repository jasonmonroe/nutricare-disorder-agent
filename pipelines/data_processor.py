# pipelines/data_processor.py
from __future__ import annotations

import nest_asyncio
import warnings

from notebooks.nutricare_disorder_agent import hypothetical_questions_retrieved
from storages.question_generator import QuestionGenerator
from storages.table_question_generator import TableQuestionGenerator

warnings.filterwarnings('ignore', category=DeprecationWarning)

from zipfile import ZipFile


# Vendors



# Local Libraries
from models.chroma import ChromaModel
from models.llama import LlamaModel
from models.openai import OpenAIModel

from src.config import (
    DOCUMENT_ZIP,
    DOCUMENT_FILE,
    DOCUMENT_DIR,
    SIMILARITY_SEARCH_QUERY,
    VECTOR_RESULT_CNT
)
from src.doc_handler import DocHandler
from src.eda import show_histogram

def run():
    # Config setup

    # Initialize streamlit persistent state

    openai_model = OpenAIModel()
    llm = openai_model.load_llm()
    llama = LlamaModel(llm, openai_model.embedding_model)


    # Apply the nested async loop to allow async code execution in the notebook
    nest_asyncio.apply()

    # === Document Ingestion & Processing ===

    # Unzipping the nutrition medical reference documents into the Nutritional Medical Reference folder
    # Loading the temp.zip and creating a zip object
    with ZipFile(DOCUMENT_ZIP, 'r') as zip_handle:
        # Extracting specific file in the zip into a specific location.
        zip_handle.extract(
            DOCUMENT_FILE,
            path=DOCUMENT_DIR
        )
        zip_handle.close()

    # Load Documents handle
    doc_handle = DocHandler(llama.parser)
    doc_handle.show_tables()


    # === Vectorization & Storage === #

    # Create vector storage for nutritional information
    chroma_db = ChromaModel({
        'llm': llm,
        'embedding_model': openai_model.embedding_model,
        'collection_name': 'nutritional'
    })

    semantic_chunks = chroma_db.get_semantic_chunks(doc_handle.folder_path)
    document_chunks = doc_handle.get_semantic_chunks(semantic_chunks)

    # Show Histogram
    show_histogram(document_chunks)

    # Text chunking using semantic chunker
    chroma_db.add_semantic_documents(document_chunks)

    # Perform similarity search in the vectorstore
    doc_handle.documents = chroma_db.semantic_storage.similarity_search(
        query=SIMILARITY_SEARCH_QUERY,
        k=VECTOR_RESULT_CNT
    )

    doc_handle.show_documents()
    chroma_db.query_questions()

    # Get hypothetical questions and add them to the vector storage
    questions = QuestionGenerator({
        'llm': llm,
        'doc_handle': doc_handle,
        'embedding_model': openai_model.embedding_model,
        'collection_name': 'hypothetical_questions',
        'document_content_description': "Hypothetical Questions for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization"
        #'semantic_chunks': document_chunks
    })

    hypothetical_questions = questions.get_hypothetical_questions(document_chunks)
    doc_handle.show_sample(hypothetical_questions, questions.collection_name.title())
    chroma_db.add_vector_documents(hypothetical_questions)

    # Get table hypothetical questions and add them to the vector storage
    table_questions = TableQuestionGenerator({
        'llm': llm,
        'doc_handle': doc_handle,
        'embedding_model': openai_model.embedding_model,
        'collection_name': 'table_hypothetical_questions',
        #'tables': doc_handle.tables

    })

    table_hypothetical_questions = table_questions.get_hypothetical_questions(doc_handle.page_texts, doc_handle.tables)
    doc_handle.show_sample(table_hypothetical_questions, table_questions.collection_name.title())
    chroma_db.add_vector_documents(table_hypothetical_questions)

    # Backup documents

    # Sample a random user query using hypothetical retriever
    chroma_db.query_questions(is_hyp=True)




def retrieve():
    pass

def store():
    pass