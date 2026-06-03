# main.py

######################## WRITE YOUR CODE HERE  #########################
#
# app.py
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

# Import necessary libraries
import os  # Interacting with the operating system (reading/writing files)
import chromadb  # High-performance vector database for storing/querying dense vectors
import nest_asyncio
import json  # Parsing and handling JSON data
import time
import zipfile

from dotenv import load_dotenv
from src.config import AI_TITLE, EXIT_CMD  # Loading environment variables from a .env file
load_dotenv()

# LangChain imports
from langchain_core.documents import Document  # Document data structures
from langchain_core.runnables import RunnablePassthrough  # LangChain core library for running pipelines
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser  # String output parser
from langchain.prompts import ChatPromptTemplate  # Template for chat prompts
from langchain.chains.query_constructor.base import AttributeInfo  # Base classes for query construction
from langchain.retrievers.self_query.base import SelfQueryRetriever  # Base classes for self-querying retrievers
from langchain.retrievers.document_compressors import LLMChainExtractor, CrossEncoderReranker  # Document compressors
from langchain.retrievers import ContextualCompressionRetriever  # Contextual compression retrievers
from langchain_core.prompts import ChatPromptTemplate as CoreChatPromptTemplate

# LangChain community & experimental imports
from langchain_community.vectorstores import Chroma  # Implementations of vector stores like Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader  # Document loaders for PDFs
from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # Cross-encoders from HuggingFace
from langchain_experimental.text_splitter import SemanticChunker  # Experimental text splitting methods
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter  # Recursive splitting of text by characters
)
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LlamaParse & LlamaIndex imports
from llama_parse import LlamaParse  # Document parsing library
from llama_index.core import Settings, SimpleDirectoryReader  # Core functionalities of the LlamaIndex

# LangGraph import
from langgraph.graph import StateGraph, END, START  # State graph for managing states in LangChain

# Pydantic import
from pydantic import BaseModel  # Pydantic for data validation

# Typing imports
from typing import Dict, List, Tuple, Any, TypedDict  # Python typing for function annotations

# Other utilities
import numpy as np  # Numpy for numerical operations

np.float_ = np.float64

from groq import Groq
from mem0 import MemoryClient
import streamlit as st
from datetime import datetime, UTC

from src.utils import show_datetime, start_timer, show_timer



def run_main_pipeline():
    pass

# --- DECLARE NUTRITION DISORDER AGENT
def nutrition_disorder_streamlit():
    """
    A Streamlit-based UI for the Nutrition Disorder Specialist Agent.
    """
    st.title(f"{AI_TITLE}")
    st.markdown("<hr style='margin: 0'>", unsafe_allow_html=True)
    st.info(body="""
    Welcome! I'm your **Dedicated AI Nutrition Agent**.
    I specialize in providing information about **nutrition disorders**, including **symptoms, causes, treatment options, and preventative measures.**
    I'm ready to answer your health-related questions.
    """, icon="📢")

    st.warning(body=f"Type **{EXIT_CMD}** at anytime to end the conversation.", icon="🪬") # Used EXIT_CMD constant here

    # Initialize the session state for chat history and user_id if they don't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    # Login form: Only if the user is not logged in
    if st.session_state.user_id is None:
        with st.form("login_form", clear_on_submit=True):
            st.write(f"Session Start: {show_datetime()}")
            user_id = st.text_input("Agent: Please enter your name to begin:").strip()

            # Don't let the username themselves a keyword
            if EXIT_CMD in user_id:
                st.error(body="You cannot name yourself a keyword.", icon="🚨")
                st.stop()

            submit_button = st.form_submit_button("Login")

            if submit_button and user_id:
                st.session_state.user_id = user_id
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Agent: Welcome, {user_id}! How can I help you with nutrition disorders today?"
                })
                st.session_state.login_submitted = True  # Set flag to trigger rerun

        if st.session_state.get("login_submitted", False):
            st.session_state.pop("login_submitted")
            st.rerun()
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input with custom placeholder text.  The user-facing prompt
        user_query = st.chat_input(f"Agent: Ask your question here, {st.session_state.user_id} (or '{EXIT_CMD}')...")

        if user_query:
            if user_query.lower() == EXIT_CMD:
                st.session_state.chat_history.append({"role": "user", "content": EXIT_CMD})

                with st.chat_message("User"):
                    st.write(EXIT_CMD)

                goodbye_msg = "Agent: Goodbye! Feel free to return if you have more questions about nutrition disorders."
                st.session_state.chat_history.append({"role": "assistant", "content": goodbye_msg})

                with st.chat_message("assistant"):
                    st.write(goodbye_msg)

                st.session_state.user_id = None
                st.rerun()
                return

            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("User"):
                st.write(f"{st.session_state.user_id}: {user_query}")

            thinking = st.empty()
            thinking.info(body="Thinking. . .", icon="🤔")

            # Filter input using Llama Guard
            filtered_result = filter_input_with_llama_guard(user_query)
            filtered_result = filtered_result.replace("\n", " ")  # Normalize the result

            # Check if input is safe based on allowed statuses
            if filtered_result in ["SAFE", "BYPASS_SAFE", ""]:
                try:

                    # Get the cached chatbot instance
                    st.session_state.chatbot = get_chatbot_instance()
                    response = st.session_state.chatbot.handle_customer_query(
                        st.session_state.user_id,
                        user_query
                    )

                    with st.chat_message("assistant"):
                        st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error while processing your query. Please try again."
                    error_str = f"Error: {str(e)}"
                    with st.chat_message("assistant"):
                        st.error(body=error_str, icon="😩")
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg + " " + error_str})

            else:
                # Unsafe queries are handled here!
                inappropriate_msg = "I apologize, but I cannot process that input as it may be inappropriate. Please try again."
                with st.chat_message("assistant"):
                    st.warning(body=inappropriate_msg, icon="🤬")

                st.session_state.chat_history.append({"role": "assistant", "content": inappropriate_msg})

            thinking.empty()


if __name__ == 'main':
    # Start Program
    run_main_pipeline()
    # End Program

    # --- RUN THE AI AGENT --- #
    start_time = start_timer()
    nutrition_disorder_streamlit()
    show_timer(start_time)







