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

global workflow_app



# Import necessary libraries


# Python Libraries
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from __future__ import annotations

# Vendor Libraries
import numpy as np
np.float_ = np.float64

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

# Local Libraries
from models.chroma import ChromaModel
from models.llama import LlamaModel
from models.openai import OpenAIModel

from pipelines.agent import build as run_build_agent_pipeline, start as run_start_agent_pipeline
from pipelines.data_processor import run as run_data_retrieval_pipeline
from pipelines.huggingface import Huggingface
from pipelines.streamlit_app import StreamLitApp

from src.utils import get_run_id, show_title_banner, start_timer, show_timer


def run_huggingface_deployment_pipeline():

    # Write Dockerfile

    # Load Huggingface
    hf = Huggingface()
    hf.deploy()

def run_streamlit_pipeline(llama: LlamaModel):

    # Note: agents must be build and started before you can run this!
    streamlit = StreamLitApp(llama)
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
    openai_model = OpenAIModel()

    # Create vector storage for nutritional information
    chroma_db = ChromaModel({
        'llm': openai_model.llm,
        'embedding_model': openai_model.embedding_model,
        'collection_name': 'nutritional'
    })

    llama = LlamaModel(openai_model.llm, openai_model.embedding_model)

    dataset = {
        'chrome_db': chroma_db,
        'llama': llama,
        'openai_model': openai_model,
    }

    for arg in args:
        if arg == 'data' or arg is None:
            run_data_retrieval_pipeline(dataset)
        if arg == 'build':
            workflow_app = run_build_agent_pipeline(dataset)
            dataset['workflow_app'] = workflow_app
        if arg == 'start':
            dataset['log'] = args['log']
            run_start_agent_pipeline(dataset)
        if arg == 'deploy':
            run_huggingface_deployment_pipeline()
        if arg == 'run':
            run_streamlit_pipeline(dataset['llama'])

    show_timer(start_time)
    print(f'\n----- ⏱️ END RUN ID: {run_id} ⏱️ -----')
