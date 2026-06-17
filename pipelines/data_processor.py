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
from src.eda import show_histogram

from storages.question_generator import QuestionGenerator
from storages.table_question_generator import TableQuestionGenerator

from src.constants import (
    DOCUMENT_CHUNK_TEXT_BATCH_SIZE,
    DOCUMENT_CHUNK_BATCH_SIZE,
    I_CHECKMARK,
    I_DIR,
    I_DISK,
    I_FLAG,
    I_RUNNING,
    I_INFO,
    CHROMA_VECTORS_DIR,
    PROMPT_TABLE_QUESTION_GENERATOR,
    PROMPT_QUESTION_GENERATOR
)
from src.doc_handler import DocHandler


def run(dataset: dict) -> None:
    """
    Run the data retrieval pipeline.
    Section 1: Comprehensive Data Parsing and Preparation for Efficient Nutritional Information Retrieval
    
    # Professional Version:
    # https://www.merckmanuals.com/professional/nutritional-disorders/nutrition-general-considerations/overview-of-nutrition

    # Consumer Version:
    # https://www.merckmanuals.com/home/disorders-of-nutrition/overview-of-nutrition/overview-of-nutrition

    :param dataset:
    :return:
    """
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    print(f'\n# --- {I_RUNNING} Running data processor pipeline {I_RUNNING} --- #')

    # Pluck all the datasets needed to run this
    llama = dataset.get('llama')
    chroma_db = dataset.get('chroma_db')
    data_refresh = dataset.get('refresh')
    print(f'{I_INFO}  Refresh flag is {data_refresh}.')

    # @todo - delete lines 60-64 when done testing
    if not data_refresh:
        print('Data refresh flag is false!')
        import sys
        sys.exit(1)

    # Apply the nested async loop to allow async code execution in the notebook.
    nest_asyncio.apply()

    semantic_count = chroma_db.get_semantic_count()
    if semantic_count > 0 and not data_refresh:
        print(f"✅ Semantic collection already has {semantic_count} documents — skipping ingestion.")

        doc_handle = DocHandler(llama.parser, skip_parse=True)
        document_chunks = []
    else:
        print('\n# --- Document Ingestion & Processing --- #')

        # Load Documents handle
        doc_handle = DocHandler(llama.parser)
        doc_handle.show_tables()

        # Create vector storage for nutritional information
        semantic_chunks = chroma_db.get_semantic_chunks(doc_handle.folder_path)
        document_chunks = doc_handle.get_semantic_chunks(semantic_chunks)
        chroma_db.add_semantic_documents(document_chunks) # @todo - erroring here!

    # Show Histogram
    # @todo - show_histogram(document_chunks)

    # Perform similarity search in the vectorstore
    doc_handle.documents = chroma_db.get_documents()
    doc_handle.show_documents()

    # Use structured receiver when quering all/random questions
    chroma_db.query_questions(is_hyp=False, pluck=random.choice([True, True, True, False]))
    chroma_dataset = chroma_db.export()

    """
    return {
            'collection_name': self.collection_name,
            #'document_content_description': self.document_content_description,
            'embedding_model': self.embedding_model,
            'force_rebuild': self.force_rebuild,
            'llm': self.llm,
            #'metadata_info': self.metadata_info,
        }
    """

    # --- Hypothetical Questions --- #

    # Get hypothetical questions and add them to the vector storage.
    # Create a merged dataset for questions.
    questions_dataset = {
        'batch_size': DOCUMENT_CHUNK_TEXT_BATCH_SIZE, #DOCUMENT_CHUNK_BATCH_SIZE,
        'collection_name': 'hypothetical_questions',
        'doc_handle': doc_handle,
        'prompt': PROMPT_QUESTION_GENERATOR,
        'title': 'Hypothetical Questions'
    }
    
    dataset = {**chroma_dataset, **questions_dataset}
    print(f'new dataset for questions:{dataset}')
    questions = QuestionGenerator(dataset, chroma_db)

    semantic_count_questions = questions.get_semantic_count()
    print(f'{I_INFO}  Existing Questions: {semantic_count_questions}')

    if semantic_count_questions > 0 and not data_refresh:
        print(f"✅ Hypothetical questions collection already has {semantic_count_questions} documents — skipping question generation.")
    elif document_chunks is not None or data_refresh:
        print(f"\nGenerating new {questions_dataset.get('title')}...")

        hypothetical_questions_doc = questions.get_hypothetical_questions(document_chunks)
        questions.add_semantic_documents(hypothetical_questions_doc)
        doc_handle.show_sample(hypothetical_questions_doc, questions.collection_name.title())
        chroma_db.add_vector_documents(hypothetical_questions_doc)
    else:
        print(f"{I_FLAG} Cannot generate hypothetical questions: document chunks unavailable.")

    # --- Hypothetical Table Questions --- #

    # Get table hypothetical questions and add them to the vector storage
    table_questions_dataset = {
        #'batch_size': DOCUMENT_CHUNK_TEXT_BATCH_SIZE,
        'collection_name': 'table_hypothetical_questions',
        'doc_handle': doc_handle,
        'prompt': PROMPT_TABLE_QUESTION_GENERATOR,
        'title': 'Hypothetical Table Questions'
    }

    dataset = {**chroma_dataset, **table_questions_dataset}
    print(f'new dataset for table questions:{dataset}')
    table_questions = TableQuestionGenerator(dataset, chroma_db)

    semantic_count_table_questions = table_questions.get_semantic_count()
    print(f'{I_INFO}  Existing Table Questions: {semantic_count_table_questions}')
    
    if semantic_count_table_questions > 0 and not data_refresh:
        print(f"✅ Hypothetical table questions collection already has {semantic_count_table_questions} documents — skipping question generation.")
    else:
    
        print(f"\nGenerating new {table_questions_dataset.get('title')}...")

        table_hypothetical_questions_doc = table_questions.get_hypothetical_questions(doc_handle.page_texts, doc_handle.tables)
        table_questions.add_semantic_documents(table_hypothetical_questions_doc)
        doc_handle.show_sample(table_hypothetical_questions_doc, table_questions.collection_name.title())
        chroma_db.add_vector_documents(table_hypothetical_questions_doc)

    # --- Backup documents to Google Drive --- #
    # backup_docs()
    # --- Backup documents to Google Drive --- #

    # Sample a random user query using hypothetical retriever
    # Note: To randomly pluck a question set pluck param to True
    chroma_db.query_questions(is_hyp=True, pluck=random.choice([True, True, True, False]))

    print(f'\n# --- {I_RUNNING} Completed data processor pipeline {I_RUNNING} --- #')


def backup_docs():
    import os 
    import shutil
    
    # Define source and destination paths for vector storage
    source_path = CHROMA_VECTORS_DIR  # Complete the code to define the path to your vectorstore directory
    destination_path = 'backup_' + CHROMA_VECTORS_DIR  # Complete the code to define the destination path in your Drive
    
    # Copy the directory to Google Drive
    try:
        shutil.copytree(source_path, destination_path)
        print(f"{I_CHECKMARK} {I_DISK} Successfully copied '{source_path}' to '{destination_path}'")
    
    except FileExistsError:
        print(f"Directory '{destination_path}' already exists. Skipping copy.")
    
    except Exception as e:
        print(f"{I_FLAG} Error copying directory: {e}")
    
    # Verify if the directory was copied successfully
    if os.path.exists(destination_path):
        print(f"{I_DISK} {source_path} directory exists on your hard drive.")  # Complete the code to confirm the directory name
    
    else:
        print(f"{source_path} directory was not copied to your source path.")  # Complete the code to confirm the directory name
        os.makedirs(destination_path, exist_ok=True)
        print(f"Making the directory {I_DIR}{destination_path} now...")
