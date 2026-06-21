from __future__ import annotations

import sys

# pipelines/data_processor.py

# +-------------------------+
# |     DATA PROCESSING     |
# +-------------------------+

# Python Libraries
import nest_asyncio
import random
import warnings

from langchain_core.documents import Document
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

        doc_handle = DocHandler(llama.parser, skip_parse=True)
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

    # Show Histogram
    # @todo - show_histogram(document_chunks)

    # Perform similarity search in the vectorstore
    doc_handle.documents = chroma_db.get_documents()

    # Show (5) similarity searched documents
    doc_handle.show_documents()

    # Use structured receiver when quering all/random questions
    chroma_db.query_questions(is_hyp=False, pluck=random.choice([True]))



    # --- Hypothetical Questions --- #
    #_process_questions(doc_handle, chroma_db, document_chunks, data_refresh)
    #print('### TERMINATE ###')
    #sys.exit(1)
    # --- Hypothetical Table Questions --- #
    _process_table_questions(doc_handle, chroma_db, data_refresh)

    # --- Backup documents to a third party storage system --- #
    #_backup_docs()
    # --- Backup documents to a third party storage system --- #

    # Sample a random user query using hypothetical retriever
    # ℹ️ Note: To randomly pluck a question set pluck param to True
    chroma_db.query_questions(is_hyp=True, pluck=random.choice([True, True, True, False]))

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


    #print('setting data_refresh to true to force a generation of questions!')
    #data_refresh = True # @todo- remove later
    #document_chunks = [Document(id='b09d5f6e141e4c57b4e7343df0199f06', metadata={'source': 'data/the-merck-manual/nutritional-disorders.pdf', 'document_domain': 'medical', 'creator': 'PyPDF', 'checksum': 'cd0cb390b21eb9cc1f66365df667c4e2679738401762c23c2220b815277a1e56', 'doc_type': 'semantic_chunk', 'total_pages': 56, 'doc_id': 'b09d5f6e141e4c57b4e7343df0199f06', 'generation_model': 'llama-3.3-70b-versatile', 'creationdate': '2026-06-17', 'page_label': '18', 'moddate': '2025-02-24T08:38:37+00:00', 'type': 'Document', 'text_id': 'parent__nutritional-disorders_pdf__p17__text', 'version': '19th Edition', 'page': 17, 'filename': 'nutritional-disorders.pdf', 'utc_datetime': '2026-06-21 03:07:11.398298+00:00', 'producer': 'iLovePDF'}, page_content='Muscle strength\n indirectly reflects increases in lean body mass. It can be measured quantitatively, by\nhand-grip dynamometry, or electrophysiologically (typically by stimulating the ulnar nerve with an\nelectrode).'), Document(id='9e77f50cd78b4d5f89a183a063c462a4', metadata={'page_label': '18', 'filename': 'nutritional-disorders.pdf', 'creationdate': '2026-06-17', 'generation_model': 'llama-3.3-70b-versatile', 'page': 17, 'checksum': '03e3ff006f452b070965c6da0b0c4fc9692d0ac398b71edef354afa17ec073c0', 'total_pages': 56, 'text_id': 'parent__nutritional-disorders_pdf__p17__text', 'moddate': '2025-02-24T08:38:37+00:00', 'source': 'data/the-merck-manual/nutritional-disorders.pdf', 'utc_datetime': '2026-06-21 03:07:11.398200+00:00', 'producer': 'iLovePDF', 'doc_id': '9e77f50cd78b4d5f89a183a063c462a4', 'type': 'Document', 'creator': 'PyPDF', 'version': '19th Edition', 'doc_type': 'semantic_chunk', 'document_domain': 'medical'}, page_content='Table 3-1\n). Assessing Response to Nutritional Support\nThere is no gold standard to assess response. Clinicians commonly use indicators of lean body mass\nsuch as the following:\n• Body mass index (BMI)\n• Body composition analysis\n• Body fat distribution (see pp. 11\n and \n58\n)\nNitrogen balance, response to skin antigens, muscle strength measurement, and indirect calorimetry can\nalso be used.'), Document(id='672a9046a13b4239a8a6063d5aa1c958', metadata={'generation_model': 'llama-3.3-70b-versatile', 'type': 'Document', 'producer': 'iLovePDF', 'moddate': '2025-02-24T08:38:37+00:00', 'creator': 'PyPDF', 'doc_type': 'semantic_chunk', 'doc_id': '672a9046a13b4239a8a6063d5aa1c958', 'checksum': 'e5f6c586edb85661744af95db3180109c22e72586587bd2940a3c017e605a106', 'document_domain': 'medical', 'filename': 'nutritional-disorders.pdf', 'version': '19th Edition', 'creationdate': '2026-06-17', 'source': 'data/the-merck-manual/nutritional-disorders.pdf', 'text_id': 'parent__nutritional-disorders_pdf__p4__text', 'total_pages': 56, 'utc_datetime': '2026-06-21 03:07:11.394385+00:00', 'page': 4, 'page_label': '5'}, page_content='Body mass index (BMI)—weight(kg)/height(m)\n2\n, which adjusts weight for height (see\nTable 6-2\n on p. 59\n), is more accurate than height and weight tables. There are standards for growth and\nweight gain in infants, children, and adolescents (see p.'), Document(id='802728e0ccf5481eb042eb7f35fddd57', metadata={'filename': 'nutritional-disorders.pdf', 'creationdate': '2026-06-17', 'doc_id': '802728e0ccf5481eb042eb7f35fddd57', 'document_domain': 'medical', 'doc_type': 'semantic_chunk', 'page': 8, 'utc_datetime': '2026-06-21 03:07:11.395719+00:00', 'type': 'Document', 'moddate': '2025-02-24T08:38:37+00:00', 'version': '19th Edition', 'generation_model': 'llama-3.3-70b-versatile', 'total_pages': 56, 'checksum': '226a55e8b0cb35e146d88c7337aab02d769b83f0b23ed310b2ad5fe820a5d1ea', 'creator': 'PyPDF', 'text_id': 'parent__nutritional-disorders_pdf__p8__text', 'source': 'data/the-merck-manual/nutritional-disorders.pdf', 'producer': 'iLovePDF', 'page_label': '9'}, page_content="2-1. Mini nutritional assessment.]\nThe mid upper arm muscle area estimates lean body mass. This area is derived from the triceps skinfold\nthickness (TSF) and mid upper arm circumference. Both are measured at the same site, with the patient's\nright arm in a relaxed position.")]

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
