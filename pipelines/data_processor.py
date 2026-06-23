from __future__ import annotations
# pipelines/data_processor.py

# +-------------------------+
# |     DATA PROCESSING     |
# +-------------------------+

# Python Libraries
import nest_asyncio
import random
import warnings

# Local Libraries
from models.chroma import ChromaModel
from src.eda import show_histogram
from src.utils import is_jupyter, show_banner
from storages.question_generator import QuestionGenerator
from storages.table_question_generator import TableQuestionGenerator

from src.constants import (
    DOCUMENT_DIR_PERM,
    DOCUMENT_FILEPATH,
    I_CHECKMARK,
    I_DIR,
    I_DISK,
    I_FLAG,
    I_RUNNING,
    I_INFO
)

from src.doc_handler import DocHandler
from src.model_config import config

def run(dataset: dict) -> None:
    """
    Run the data retrieval pipeline.
    Section 1: Comprehensive Data Parsing and Preparation for Efficient Nutritional Information Retrieval
    
    # Professional Version:
    # https://www.merckmanuals.com/professional/nutritional-disorders/nutrition-general-considerations/overview-of-nutrition

    # Consumer Version:
    # https://www.merckmanuals.com/home/disorders-of-nutrition/overview-of-nutrition/overview-of-nutrition
    
    Note: ❗ The Merck Manual of Diagnosis & Therapy, 19th Edition is a highly copyrighted commercial work owned by
    Merck & Co., Inc.
    Please do not make the full manual public or risk a DMCA takedown request.

    :param dataset:
    :return:
    """

    show_banner(f'{I_RUNNING} Running data processor pipeline {I_RUNNING}')

    if not dataset.get('log_debug'):
        warnings.filterwarnings('ignore', category=DeprecationWarning)

    # Pluck all the datasets needed to run this
    llama = dataset.get('llama')
    chroma_db = dataset.get('chroma_db')
    data_refresh = dataset.get('refresh')
    print(f'{I_INFO}  Refresh flag is {data_refresh}.')

    # Apply the nested async loop to allow async code execution in the notebook.
    if is_jupyter():
        nest_asyncio.apply()

    semantic_count = chroma_db.get_semantic_count()
    if semantic_count > 0 and not data_refresh:
        print(f"✅ Semantic collection already has {semantic_count} documents — skipping ingestion.")

        doc_handle = DocHandler(llama.parser, skip_parse=random.choice([True, False]))
        document_chunks = []
    else:
        print('\n# --- Document Ingestion & Processing --- #')

        # Load Documents handle
        doc_handle = DocHandler(llama.parser)
        doc_handle.show_tables()

        # Create vector storage for nutritional information
        semantic_chunks = chroma_db.get_semantic_chunks(DOCUMENT_FILEPATH)
        document_chunks = doc_handle.get_semantic_chunks(semantic_chunks)
        doc_handle.document_chunks = document_chunks
        chroma_db.add_semantic_documents(document_chunks)

        print(f'\nDocument Chunks: {doc_handle.count_document_chunks()}.')

    # Show Histogram
    show_histogram(document_chunks)

    # Perform similarity search in the vectorstore
    doc_handle.documents = chroma_db.get_documents()

    # Show (5) similarity searched documents
    doc_handle.show_documents()

    # Use structured receiver when quering all/random questions
    chroma_db.query_questions(is_hyp=False, pluck=random.choice([True]))

    # --- Hypothetical Questions --- #
    _process_questions(doc_handle, chroma_db, document_chunks, data_refresh)

    # --- Hypothetical Table Questions --- #
    _process_table_questions(doc_handle, chroma_db, data_refresh)

    # --- Backup documents to a third party storage system --- #
    _backup_docs()
    # --- Backup documents to a third party storage system --- #

    # Sample a random user query using hypothetical retriever
    # ℹ️ Note: To randomly pluck a question set pluck param to True
    chroma_db.query_questions(is_hyp=True, pluck=random.choice([True, False]))

    print(f'\n# --- {I_RUNNING} Completed data processor pipeline {I_RUNNING} --- #')


def _process_questions(doc_handle: DocHandler, chroma_db: ChromaModel, document_chunks: list, data_refresh: bool):

    # Get hypothetical questions and add them to the vector storage.
    questions_dataset = {
        'batch_size': config.DOCUMENT_CHUNK_TEXT_BATCH_SIZE,
        'doc_handle': doc_handle,
        'doc_type': 'hypothetical_questions',
        'prompt': config.PROMPT_QUESTION_GENERATOR,
    }

    dataset = _merge_datasets(chroma_db, questions_dataset)
    questions = QuestionGenerator(dataset, chroma_db)
    semantic_questions_count = questions.get_semantic_count()
    title = f'{questions.collection_name.capitalize()} {questions.title}'

    show_banner(title, f'{I_INFO}  Existing Questions: {semantic_questions_count}')

    if semantic_questions_count > 0 and not data_refresh:
        print(f"✅ {title} collection already has {semantic_questions_count} documents — skipping question generation.")
    elif document_chunks is not None or data_refresh:
        print(f"\nGenerating new {title}...")

        hypothetical_questions_doc = questions.get_hypothetical_questions(document_chunks)
        doc_handle.show_sample(hypothetical_questions_doc, title)

        questions.add_semantic_documents(hypothetical_questions_doc)
        questions.add_vector_documents(hypothetical_questions_doc)
    else:
        print(f"{I_FLAG} Cannot generate hypothetical questions: document chunks unavailable.")


def _process_table_questions(doc_handle: DocHandler, chroma_db: ChromaModel, data_refresh: bool):

    # Get table hypothetical questions and add them to the vector storage
    table_questions_dataset = {
        'doc_handle': doc_handle,
        'doc_type': 'table_hypothetical_questions',
        'prompt': config.PROMPT_TABLE_QUESTION_GENERATOR,
    }

    dataset = _merge_datasets(chroma_db, table_questions_dataset)
    table_questions = TableQuestionGenerator(dataset, chroma_db)
    semantic_table_questions_count = table_questions.get_semantic_count()
    title = f'{table_questions.collection_name.capitalize()} {table_questions.title}'

    show_banner(title, f'{I_INFO}  Existing {title}: {semantic_table_questions_count}')
    
    if semantic_table_questions_count > 0 and not data_refresh:
        print(f"✅ {title} collection already has {semantic_table_questions_count} documents — skipping question generation.")
    else:
        print(f"\nGenerating new {title}...")

        table_hypothetical_questions_doc = table_questions.get_hypothetical_questions(doc_handle.page_texts, doc_handle.tables)
        doc_handle.show_sample(table_hypothetical_questions_doc, title)

        table_questions.add_semantic_documents(table_hypothetical_questions_doc)
        table_questions.add_vector_documents(table_hypothetical_questions_doc)


def _merge_datasets(chroma_db: ChromaModel, dataset: dict) -> dict:
    # Create a merged dataset for questions.
    chroma_dataset = chroma_db.export()
    return {**chroma_dataset, **dataset}


def _backup_docs():
    import os 
    import shutil

    show_banner(f'{I_DISK} Backing up documents {I_DISK}')
    
    # Define source and destination paths for vector storage
    # Complete the code to define the path to your vectorstore directory.
    # Complete the code to define the destination path in your Drive.
    source_path = config.CHROMA_VECTORS_DIR
    destination_path = '_backups/' + config.CHROMA_VECTORS_DIR
    
    # Copy the directory to Google Drive
    try:
        shutil.copytree(source_path, destination_path)
        print(f"{I_CHECKMARK} Successfully copied '{source_path}' to '{destination_path}'")
    
    except FileExistsError:
        print(f"Directory '{destination_path}' already exists. Skipping copy.")
    
    except Exception as e:
        print(f"{I_FLAG} Error copying directory: {e}")
    
    # Verify if the directory was copied successfully
    if os.path.exists(destination_path):
        print(f"{I_DIR} {source_path} directory exists on your hard drive.")
    
    else:
        print(f"{source_path} directory was not copied to your source path.")
        os.makedirs(destination_path, exist_ok=True)
        os.chmod(destination_path, DOCUMENT_DIR_PERM)
        print(f"Making the directory {I_DIR}{destination_path} now with permissions: {DOCUMENT_DIR_PERM}...")
