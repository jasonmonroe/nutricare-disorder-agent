from __future__ import annotations

# pipelines/data_processor.py

# +-------------------------+
# |     DATA PROCESSING     |
# +-------------------------+
#
# Source: https://www.merckmanuals.com/professional/nutritional-disorders/nutrition-general-considerations/overview-of-nutrition
# Source: https://www.merckmanuals.com/home/disorders-of-nutrition/overview-of-nutrition/overview-of-nutrition

import nest_asyncio
import warnings
from zipfile import ZipFile

# Local Libraries
from storages.question_generator import QuestionGenerator
from storages.table_question_generator import TableQuestionGenerator

from src.config import (
    DOCUMENT_CHUNK_TEXT_BATCH_SIZE,
    DOCUMENT_CHUNK_BATCH_SIZE,
    DOCUMENT_DIR,
    DOCUMENT_FILE,
    DOCUMENT_ZIP,
    I_DB,
    SIMILARITY_SEARCH_QUERY,
    VECTOR_RESULT_CNT, I_DISK,
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

    print(f'# --- {I_DB} Running data processor pipeline {I_DB} --- #')

    # Pluck all the datasets needed to run this
    llama = dataset['llama']
    chroma_db = dataset['chroma_db']

    # Apply the nested async loop to allow async code execution in the notebook
    nest_asyncio.apply()

    # === Document Ingestion & Processing ===

    # Load Documents handle
    doc_handle = DocHandler(llama.parser)
    doc_handle.show_tables()

    # === Vectorization & Storage === #

    # Create vector storage for nutritional information
    semantic_chunks = chroma_db.get_semantic_chunks(doc_handle.folder_path)
    document_chunks = doc_handle.get_semantic_chunks(semantic_chunks)

    # Show Histogram
    show_histogram(document_chunks)

    # Text chunking using semantic chunker
    # Note: Is hyp = False
    # DEBUG: show Chroma client and db directory contents before writing
    try:
        import os, stat
        client = getattr(chroma_db, 'chromadb_client', None)
        print(f"DEBUG: chromadb_client = {client}")
        if client is not None:
            try:
                print('DEBUG: client.get_settings():', client.get_settings())
                print('DEBUG: client.list_collections():', client.list_collections())
            except Exception as e:
                print('DEBUG: could not query client settings:', e)

        print('DEBUG: cwd =', os.getcwd())
        if os.path.exists('db'):
            print('DEBUG: db/ contents:')
            for p in os.listdir('db'):
                full = os.path.join('db', p)
                try:
                    mode = oct(os.stat(full).st_mode & 0o777)
                except Exception:
                    mode = 'n/a'
                print(f"  - {full} (mode={mode})")
        else:
            print('DEBUG: db/ does not exist')
    except Exception as _:
        pass

    chroma_db.add_semantic_documents(document_chunks)

    # Perform similarity search in the vectorstore
    doc_handle.documents = chroma_db.semantic_storage.similarity_search(
        query=SIMILARITY_SEARCH_QUERY,
        k=VECTOR_RESULT_CNT
    )

    doc_handle.show_documents()

    # Use structured receiver when quering all/random questions
    chroma_db.query_questions(is_hyp=False, pluck=False)

    # Get hypothetical questions and add them to the vector storage
    document_content_desc =  DOCUMENT_DIR + ' published by the Global Nutritional Health Organization'

    chroma_dataset = chroma_db.export()
    questions_dataset = {
        'batch_size': DOCUMENT_CHUNK_BATCH_SIZE,
        'collection_name': 'hypothetical_questions',
        'doc_handle': doc_handle,
        'document_content_description': 'Hypothetical Questions for ' + document_content_desc,
    }

    dataset = {**chroma_dataset, **questions_dataset}
    questions = QuestionGenerator(dataset)
    hypothetical_questions_doc = questions.get_hypothetical_questions(document_chunks)
    doc_handle.show_sample(hypothetical_questions_doc, questions.collection_name.title())
    chroma_db.add_vector_documents(hypothetical_questions_doc)

    # Get table hypothetical questions and add them to the vector storage
    table_questions_dataset = {
        'batch_size': DOCUMENT_CHUNK_TEXT_BATCH_SIZE,
        'collection_name': 'table_hypothetical_questions',
        'doc_handle': doc_handle,
        'document_content_description': 'Hypothetical Table Questions for ' + document_content_desc,
    }

    dataset = {**chroma_dataset, **table_questions_dataset}
    table_questions = TableQuestionGenerator(dataset)
    table_hypothetical_questions_doc = table_questions.get_hypothetical_questions(doc_handle.page_texts, doc_handle.tables)
    doc_handle.show_sample(table_hypothetical_questions_doc, table_questions.collection_name.title())
    chroma_db.add_vector_documents(table_hypothetical_questions_doc)

    # --- Backup documents to Google Drive --- #
    __backup_docs()
    # --- Backup documents to Google Drive --- #

    # Sample a random user query using hypothetical retriever
    # Note: To randomly pluck a question set pluck param to True
    chroma_db.query_questions(is_hyp=True, pluck=False)

    print('DEBUG: Exiting data_processor:run() ...')

def __backup_docs():
    pass
