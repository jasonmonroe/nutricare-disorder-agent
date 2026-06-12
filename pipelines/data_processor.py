from __future__ import annotations

import random

# pipelines/data_processor.py

# +-------------------------+
# |     DATA PROCESSING     |
# +-------------------------+
#
# Source: https://www.merckmanuals.com/professional/nutritional-disorders/nutrition-general-considerations/overview-of-nutrition
# Source: https://www.merckmanuals.com/home/disorders-of-nutrition/overview-of-nutrition/overview-of-nutrition

import nest_asyncio
import warnings

# Local Libraries
from storages.question_generator import QuestionGenerator
from storages.table_question_generator import TableQuestionGenerator

from src.config import (
    DOCUMENT_CHUNK_TEXT_BATCH_SIZE,
    DOCUMENT_CHUNK_BATCH_SIZE,
    DOCUMENT_DIR,
    I_RUNNING
)
from src.doc_handler import DocHandler
from src.eda import show_histogram


def run(dataset: dict) -> None:
    """
    Run the data retrieval pipeline.
    Section 1: Comprehensive Data Parsing and Preparation for Efficient Nutritional Information Retrieval

    :param dataset:
    :return:
    """
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    print(f'\n# --- {I_RUNNING} Running data processor pipeline {I_RUNNING} --- #')

    # Pluck all the datasets needed to run this
    llama = dataset['llama']
    chroma_db = dataset['chroma_db']

    # Apply the nested async loop to allow async code execution in the notebook
    nest_asyncio.apply()

    existing = chroma_db.get_semantic_count()
    if existing > 0:
        print(f"✅ Semantic collection already has {existing} documents — skipping ingestion.")
        doc_handle = DocHandler(llama.parser, skip_parse=True)
        document_chunks = []
    else:
        # === Document Ingestion & Processing ===

        # Load Documents handle
        doc_handle = DocHandler(llama.parser)
        doc_handle.show_tables()

        # Create vector storage for nutritional information
        # === Vectorization & Storage === #
        semantic_chunks = chroma_db.get_semantic_chunks(doc_handle.folder_path)
        document_chunks = doc_handle.get_semantic_chunks(semantic_chunks)
        chroma_db.add_semantic_documents(document_chunks)

        # Show Histogram @todo - uncomment when ready for prod
        #show_histogram(document_chunks)

    # Perform similarity search in the vectorstore
    doc_handle.documents = chroma_db.get_documents()
    doc_handle.show_documents()

    # Use structured receiver when quering all/random questions
    chroma_db.query_questions(is_hyp=False, pluck=random.choice([True, False]))

    # Get hypothetical questions and add them to the vector storage
    document_content_desc =  DOCUMENT_DIR + ' published by the Global Nutritional Health Organization'

    # Create a merged dataset for questions.
    chroma_dataset = chroma_db.export()
    questions_dataset = {
        'batch_size': DOCUMENT_CHUNK_BATCH_SIZE,
        'collection_name': 'hypothetical_questions',
        'doc_handle': doc_handle,
        'document_content_description': 'Hypothetical Questions for ' + document_content_desc,
    }
    
    dataset = {**chroma_dataset, **questions_dataset}
    questions = QuestionGenerator(dataset)

    existing_questions = questions.get_semantic_count()
    if existing_questions > 0:
        print(f"✅ Hypothetical questions collection already has {existing_questions} documents — skipping question generation.")
    elif document_chunks is not None:
        hypothetical_questions_doc = questions.get_hypothetical_questions(document_chunks)
        questions.add_semantic_documents(hypothetical_questions_doc)
        doc_handle.show_sample(hypothetical_questions_doc, questions.collection_name.title())
        chroma_db.add_vector_documents(hypothetical_questions_doc)
    else:
        print("Cannot generate hypothetical questions: document chunks unavailable.")

    # Get table hypothetical questions and add them to the vector storage
    table_questions_dataset = {
        'batch_size': DOCUMENT_CHUNK_TEXT_BATCH_SIZE,
        'collection_name': 'table_hypothetical_questions',
        'doc_handle': doc_handle,
        'document_content_description': 'Hypothetical Table Questions for ' + document_content_desc,
    }

    dataset = {**chroma_dataset, **table_questions_dataset}
    table_questions = TableQuestionGenerator(dataset)

    existing_table_questions = table_questions.get_semantic_count()
    if existing_table_questions > 0:
        print(f"✅ Hypothetical table questions collection already has {existing_table_questions} documents — skipping question generation.")
    else:
        table_hypothetical_questions_doc = table_questions.get_hypothetical_questions(doc_handle.page_texts, doc_handle.tables)
        table_questions.add_semantic_documents(table_hypothetical_questions_doc)
        doc_handle.show_sample(table_hypothetical_questions_doc, table_questions.collection_name.title())
        chroma_db.add_vector_documents(table_hypothetical_questions_doc)

    # --- Backup documents to Google Drive --- #
  
    # --- Backup documents to Google Drive --- #

    # Sample a random user query using hypothetical retriever
    # Note: To randomly pluck a question set pluck param to True
    chroma_db.query_questions(is_hyp=True, pluck=random.choice([True, False]))

    print(f'\n# --- {I_RUNNING} Completed data processor pipeline {I_RUNNING} --- #')
