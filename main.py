from __future__ import annotations

# main.py

import os
from dotenv import load_dotenv
from sqlalchemy.sql import false

# Force load_dotenv to overwrite any existing terminal environmental variables
load_dotenv(override=True)

# --- OpenAI Base URL Settings --- #
OPENAI_API_BASE_ENV = os.environ.get("OPENAI_API_BASE")

# Defensive Safety Guard: Ensure the environment variable actually loaded
if not OPENAI_API_BASE_ENV:
    print("\n[CRITICAL ERROR] 'OPENAI_API_BASE' is missing from your .env file.")
    print("Please check your configuration files before running the pipeline.\n")
    import sys
    sys.exit(1)

# Explicitly re-bind it to guarantee LangChain background workers capture it
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE_ENV
# --- OpenAI Base URL Settings --- #

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

import sys
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

from src.config import I_INFO, I_TIMER, I_WARNING
from src.doc_handler import DocHandler
from src.utils import get_run_id, show_title_banner, start_timer, show_timer


def run_huggingface_deployment_pipeline():
    # Load Huggingface
    hf = Huggingface()
    hf.deploy()


def run_streamlit_pipeline(llama_obj: LlamaModel):
    # Note: agents must be built and started before you can run this!
    streamlit = StreamLitApp(llama_obj)
    streamlit.run()


def _parse_args(command_line_args: list[str]) -> dict:
    """
    Parse arguments present in command line.
    :param command_line_args:
    :return:
    """
    if len(command_line_args) == 0:
        print(f'{I_WARNING} No args present... exiting. {I_WARNING}')
        sys.exit(1)

    args_list = ['--build', '--data', '--deploy', '--log', '--mock', '--run', '--start']

    return {arg.strip('--'): (arg in command_line_args) for arg in args_list}


# Ensure your entry block checks against '__main__', not 'main'
if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    run_id = get_run_id()
    print(f'\n-----> {I_TIMER} START RUN ID: {run_id} {I_TIMER} <-----\n')
    start_time = start_timer()

    args = _parse_args(sys.argv[1:])
    mock = args.get('mock', False)
    log = args.get('log', False)

    show_title_banner()

    if mock:
        print(f'{I_INFO} Mock mode is turned on!')

    if log:
        logging.basicConfig(level=logging.INFO) #DEBUG
    else:
        logging.basicConfig(level=logging.INFO)

    # Wipe documents directory before Chroma is created.
    if args.get('data'):
        DocHandler.wipe_db_dir()
        
    # --- Load all models --- #
    openai_model = OpenAIModel(mock=mock)

    # Create vector storage for nutritional information
    chroma_db = ChromaModel({
        'llm': openai_model.llm,
        'embedding_model': openai_model.embedding_model,
        'collection_name': 'nutritional',
        'mock': mock
    })

    llama = LlamaModel(openai_model.llm, openai_model.embedding_model, log)

    # Create pipeline dataset.
    dataset = {
        'chroma_db': chroma_db,
        'llama': llama,
        'openai_model': openai_model,
        'mock': mock,
        'log': log
    }
    
    # Execute based on parsed flags
    if args.get('data'):
        run_data_retrieval_pipeline(dataset)

    if args.get('build'):
        workflow_app = run_build_agent_pipeline(dataset)
        dataset['workflow_app'] = workflow_app

    if args.get('start'):
        run_start_agent_pipeline(dataset)

    if args.get('deploy'):
        run_huggingface_deployment_pipeline()

    if args.get('run'):
        run_streamlit_pipeline(llama)

    show_timer(start_time)

    print(f'\n-----> {I_TIMER} END RUN ID: {run_id} {I_TIMER} <-----\n')
