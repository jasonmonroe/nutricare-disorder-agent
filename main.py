# main.py


#
# main.py app.py
# Published by Jason Monroe
# jason@jasonmonroe.com
# Date Created: 2024-11-16
# Script for AI Agent for Huggingface Space
# https://huggingface.co/spaces/jasonmonroe/smart-nutri-disorder-specialist-bot
#
# [MODULE NAME]: app.py
#
# Description:
#    A Streamlit-based AI chatbot application that acts as a "Nutrition Disorder Specialist."
#    This script performs the following key functions:
#    1.  **Document Ingestion & Processing:** Loads and parses PDF documents from a specified directory (`Nutritional Medical Reference`). It uses LlamaParse to extract text and structured data (tables).
#     2.  **Vectorization & Storage:** Chunks the processed text using semantic chunking and stores the text, along with hypothetical questions generated from the content, into a Chroma vector database. This creates a searchable knowledge base.
#     3.  **Agentic RAG Workflow:** Implements a sophisticated Retrieval-Augmented Generation (RAG) workflow using LangGraph. This workflow includes steps for query expansion, context retrieval, response generation, and self-correction loops for groundedness and precision.
#     4.  **Conversational AI:** Provides a conversational interface where users can ask questions about nutritional disorders. It uses a `NutritionBot` class that manages user sessions, conversation history (with Mem0), and interacts with the RAG agent.
#     5.  **Safety & Moderation:** Filters user input using Llama Guard to prevent inappropriate or harmful queries.
#
# Dependencies:
#     - streamlit: For the web application interface.
#     - langchain, langgraph, llama_parse, llama_index: Core libraries for the RAG pipeline and agentic workflow.
#     - chromadb: For vector storage and retrieval.
#     - openai, groq: For accessing LLMs and safety models.
#     - mem0: For managing conversational memory.
#     - dotenv: For managing environment variables.
#     - numpy, pandas: For data manipulation.
#
# Usage:
#     Run the script as a Streamlit application. The application will start a chat interface
#     where users can log in with a name and ask questions about nutritional disorders.
#

# --- IMPORT LIBRARIES
import sys
import warnings

# Import necessary libraries
import nest_asyncio
from models.chroma import ChromaModel
from models.llama import LlamaModel
from models.openai import OpenAIModel
from src.doc_handler import DocHandler

warnings.filterwarnings('ignore', category=DeprecationWarning)

from __future__ import annotations

# Vendor Libraries

import numpy as np
np.float_ = np.float64


from dotenv import load_dotenv
load_dotenv()


# === Local libraries ===

# - Pipelines
from pipelines.agent import build as run_build_agent_pipeline, start as run_start_agent_pipeline
from pipelines.data_processor import run as run_data_retrieval_pipeline
from pipelines.huggingface import Huggingface
from pipelines.streamlit_app import StreamLitApp


# - Source Files
from src.eda import show_histogram
from src.config import DOCUMENT_ZIP, SIMILARITY_SEARCH_QUERY

from src.utils import get_run_id, show_title_banner, start_timer, show_timer




# LangChain imports
from langchain.prompts import ChatPromptTemplate  # Template for chat prompts
from langchain.chains.query_constructor.base import AttributeInfo  # Base classes for query construction
from langchain.retrievers.self_query.base import SelfQueryRetriever  # Base classes for self-querying retrievers
from langchain.retrievers.document_compressors import LLMChainExtractor, CrossEncoderReranker  # Document compressors
from langchain.retrievers import ContextualCompressionRetriever  # Contextual compression retrievers

# LangChain community & experimental imports
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter  # Recursive splitting of text by characters
)
from langchain.agents import create_tool_calling_agent, AgentExecutor

# LlamaParse & LlamaIndex imports

# LangGraph import

# Pydantic import

# Typing imports


# Other utilities
#import numpy as np  # Numpy for numerical operations

#np.float_ = np.float64


# Vendors

# Local



def run_main_pipeline():





    






    # === Retrieve Context from Vector Storage ===





    # === Build Advanced Agent ===
    nutrition_bot = NutritionBot()

    # === Agentic RAG Workflow ===
    # === Conversational AI ===
    # === Safety & Moderation ===



    # === Build App ===




    # === Deploy to HuggingFace ===


    """
    Init embedding function, embedding model, llm, Settings.lmm
    nest_asyncio.apply()

    get parser


    parse content from documents with handler
    extract tables from parsed objects
    show dump of parsed doc chunks

    init chromadb client

    get semantic text splitter

    text chunking using semantic chunker

    creating and stroring docs in chroma vector store
    get semanticstore
    add docs to semantic store

    perform similarity search in the vector storage
    define metadata field info
    get structured retriever
    self retriever queries
    show queries

    create hypothecical questions prompt
    set hypothetical questions
    create hypothetical questions for table prompt
    set hypothetical questions table

    store hypothetical chunks in chroma vectorstore
    backup nutrinal vectorstore on local harddrive

    init metadata field info

    get structure hyptothetical retriever

    # build advcanced rag tool
    read config
    load chromadb and openai embeddings

    define agentic state
    show_logs = True



    retrive context from vector store
    get original docs
    add docs to vectorstore

    define AI workflow for nutrition bot

    define agentic rag queries
    show rag queries

    define llama guard railings

    define nutrition disorder agent (streamlit)

    --

    deploy app to huggingface
    load app.py
    write requirements.txt
    run docker file
    load huggingface class
    upload files


    """

    # Apply the nested async loop to allow async code execution in the notebook
    nest_asyncio.apply()




    # Now run Streamlit App

    pass

"""
Section 1: Comprehensive Data Parsing and Preparation for Efficient Nutritional Information Retrieval
"""







"""
Section 3: Getting Your App Live on Hugging Face Docker Spaces
"""

def run_huggingface_deployment_pipeline():
    hf = Huggingface()
    hf.deploy()

def run_streamlit_pipeline():

    # Note: agents must be build and started before you can run this!
    streamlit = StreamLitApp
    streamlit.run()



def _parse_args(command_line_args: list[str]):
    """
    Parse arguments presents in command line.
    :param command_line_args:
    :return:
    """
    if len(command_line_args) == 0:
        return None

    args_list = ['--data', '--build', '--start', '--deploy', '--run', '--log']

    return {arg.strip('--'): (arg in command_line_args) for arg in args_list}


if __name__ == 'main':

    run_id = get_run_id()
    print(f'\n----- ⏱️ START RUN ID: {run_id} ⏱️ -----\n')
    start_time = start_timer()

    show_title_banner()
    args = _parse_args(sys.argv[1:])

    # --- Load all models --- #

    dataset = {
        'llm': '',
        'llama': '',
        'chrome_db': '',

    }

    if args is None:
        run_data_retrieval_pipeline()
    else:
        for arg in args:
            if arg == 'data' or arg is None:
                run_data_retrieval_pipeline()
            if arg == 'build':
                run_build_agent_pipeline(args['log'])
            if arg == 'start':
                run_start_agent_pipeline()
            if arg == 'deploy':
                run_huggingface_deployment_pipeline()
            if arg == 'run':
                run_streamlit_pipeline()

    show_timer(start_time)
    print(f'\n----- ⏱️ END RUN ID: {run_id} ⏱️ -----')
