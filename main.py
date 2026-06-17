from __future__ import annotations

# main.py

"""
+--------------+
|     MAIN     |
+--------------+

Nutrition Disorder Specialist Streamlit Application.

This module implements a Streamlit-based AI chatbot application that acts as a
"Nutrition Disorder Specialist." It leverages an advanced Agentic RAG workflow
to process medical reference documents and provide grounded, safe answers.

Key Functions:
    1. Document Ingestion & Processing: Loads and parses PDF documents from a
       specified directory (`Nutritional Medical Reference`) using LlamaParse
       to extract text and structured tables.
    2. Vectorization & Storage: Chunks processed text using semantic chunking
       and stores the text, along with generated hypothetical questions, into
       a Chroma vector database.
    3. Agentic RAG Workflow: Implements a sophisticated Retrieval-Augmented
       Generation (RAG) workflow using LangGraph, including query expansion,
       context retrieval, response generation, and self-correction loops.
    4. Conversational AI: Provides a chat interface via a `NutritionBot` class
       that manages user sessions and conversation history using Mem0.
    5. Safety & Moderation: Filters user input using Llama Guard to prevent
       inappropriate or harmful queries.

Dependencies:
    - streamlit: Web application interface.
    - langchain, langgraph, llama_parse, llama_index: RAG pipeline and agentic
      workflow orchestration.
    - chromadb: Vector storage and retrieval.
    - openai, groq: LLM infrastructure and safety models.
    - mem0: Conversational memory management.
    - dotenv: Environment variable configuration.
    - numpy, pandas: Data manipulation.

Usage:
    Run the script as a Streamlit application:
        $ streamlit run app.py
    The application will launch a chat interface where users can log in with
    a name and ask questions about nutritional disorders.
"""

__author__ = "Jason Monroe (jason@jasonmonroe.com)"
__copyright__ = "Copyright November 11-26 2024, Scripts for AI Agent for Huggingface Space"
__date__ = "2024-11-16"
__version__ = "1.0.0"


# Global compilation configurations
global workflow_app

# Python Libraries
import os
import sys
from dotenv import load_dotenv

# Force load_dotenv to overwrite any existing terminal environmental variables
load_dotenv(override=True)

# --- OpenAI Base URL Settings --- #
OPENAI_API_BASE_ENV = os.environ.get("OPENAI_API_BASE")

# Defensive Safety Guard: Ensure the environment variable actually loaded
if not OPENAI_API_BASE_ENV:
    print("\n[CRITICAL ERROR] 'OPENAI_API_BASE' is missing from your .env file.")
    print("Please check your configuration files before running the pipeline.\n")
    sys.exit(1)

# Explicitly re-bind it to guarantee LangChain background workers capture it
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE_ENV

# --- OpenAI Base URL Settings --- #
if sys.version_info >= (3, 13):
    print("CRITICAL: This project requires Python 3.11 or 3.12. Python 3.13+ is not yet supported.")
    sys.exit(1)

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pydantic')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='langchain')

# Vendor Libraries
import logging
import numpy as np
np.float_ = np.float64

# Local Libraries
from models import ChromaModel, LlamaModel, OpenAIModel

from pipelines.agent import build as run_build_agent_pipeline, start as run_start_agent_pipeline
from pipelines.data_processor import run as run_data_retrieval_pipeline
from pipelines.huggingface import Huggingface 
from pipelines.streamlit_app import StreamLitApp

from src.constants import ARG_PARAMS, DEFAULT_COLL_NAME, I_BOT, I_CROSSMARK, I_SKULL, I_TIMER, I_WARNING
from src.doc_handler import DocHandler
from src.utils import get_run_id, show_title_banner, start_timer, show_timer


def _parse_args(command_line_args: list[str]) -> dict:
    """
    Parse arguments present in command line.
    :param command_line_args:
    :return:
    """
    if len(command_line_args) == 0:
        print(f'{I_WARNING} No args present... exiting. {I_WARNING}')
        sys.exit(1)

    return {arg.strip('--'): (arg in command_line_args) for arg in ARG_PARAMS}


def run_huggingface_deployment_pipeline():
    # Load Huggingface
    hf = Huggingface()
    hf.deploy()


def run_streamlit_pipeline(llama_obj: LlamaModel):
    # Note: agents must be built and started before you can run this!
    streamlit = StreamLitApp(llama_obj)
    streamlit.run()


def _check_models():
    # Safeguard block evaluation order prevents unhandled NoneType errors
    if chroma_db is None or openai_model is None or llama is None:
        print(f"{I_CROSSMARK} Core dependencies didn't load properly. Exiting system!!! {I_CROSSMARK}")
        print(f'{I_SKULL}')
        sys.exit(0)


def _check_chroma_db():
    semantic_count = chroma_db.get_semantic_count()
    vector_count = chroma_db.get_document_count()

    if semantic_count == 0 and vector_count == 0:
        print(f"{I_CROSSMARK} Out of sync! Both vector partitions are completely empty. Please run with --data first! {I_CROSSMARK}")
        raise RuntimeError("Vector database contains zero records across all internal collections.")

    if chroma_db.get_document_count() == 0:
        print(f"{I_CROSSMARK} No documents found in the vector storage. Please run with --data first! {I_CROSSMARK}")
        raise RuntimeError("Vector database is completely empty!")


def _set_logger(args):
    log = args.get('log', False)
    log_debug = args.get('log.debug', False)

    logger = logging.getLogger(__name__)
    if log:
        logging.basicConfig(level=logging.INFO) #DEBUG/INFO
    elif log_debug:
        logging.basicConfig(level=logging.DEBUG)
    
    return log


if __name__ == '__main__':

    start_time = start_timer()
    run_id = get_run_id()
    print(f'\n==== {I_BOT} START RUN ID: {run_id} {I_BOT} ====\n')
    show_title_banner()

    args = _parse_args(sys.argv[1:])
    print(f'DEBUG:args={args}')
    log = _set_logger(args)
    refresh = True if args.get('refresh') else False

    # Wipe documents directory before Chroma is created.
    if refresh:
        DocHandler.wipe_db_dir()

    # --- Load all models --- #
    openai_model = OpenAIModel()

    # Create vector storage for nutritional information
    chroma_db = ChromaModel({
        'llm': openai_model.llm,
        'embedding_model': openai_model.embedding_model,
        'collection_name': DEFAULT_COLL_NAME,
        'force_rebuild': refresh
        
    })

    llama = LlamaModel(openai_model.llm, openai_model.embedding_model, log)

    # --- Check Models --- #

    # Safeguard block evaluation order prevents unhandled NoneType errors
    _check_models()

    # Create pipeline dataset.
    dataset = {
        'chroma_db': chroma_db,
        'llama': llama,
        'openai_model': openai_model,
        'log': log,
        'refresh': refresh
    }
    
    # --- Execute based on parsed flags --- #
    
    

    if args.get('data'):
        run_data_retrieval_pipeline(dataset)

    if args.get('build'):
        workflow_app = run_build_agent_pipeline(dataset)
        dataset['workflow_app'] = workflow_app

    if args.get('start'):
        _check_chroma_db()
        run_start_agent_pipeline(dataset)

    if args.get('deploy'):
        run_huggingface_deployment_pipeline()

    if args.get('run'):
        run_streamlit_pipeline(llama)

    show_timer(start_time)

    print(f'\n==== {I_BOT} END RUN ID: {run_id} {I_BOT} ====\n')
