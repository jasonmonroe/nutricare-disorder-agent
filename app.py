
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

from dotenv import load_dotenv  # Loading environment variables from a .env file
load_dotenv()

# LangChain imports
from langchain_core.documents import Document  # Document data structures
from langchain_core.runnables import RunnablePassthrough  # LangChain core library for running pipelines
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser  # String output parser
from langchain_classic.prompts import ChatPromptTemplate  # Template for chat prompts
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate as CoreChatPromptTemplate

# LangChain community & experimental imports
from langchain_community.vectorstores import Chroma  # Implementations of vector stores like Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader  # Document loaders for PDFs
from langchain_experimental.text_splitter import SemanticChunker  # Experimental text splitting methods
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
#from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
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

# --- DEFINE CONFIGURATIONS AND CONSTANTS
# Note: os.getenv() are the secrets defined in the Huggingface.co settings page.
# os.getenv() is for READING a variable from the operating system's environment.

# Hugging Face
# see: https://hugginface.co
# see: Model -> https://huggingface.co/jasonmonroe/smart-nutri-disorder-specialist-model
# see: Space -> https://huggingface.co/jasonmonroe/smart-nutri-disorder-specialist-bot
# Note: Use your own Huggingface Repo ID
HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

# Groq
# see: https://www.groq.com/
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Llama
# see: https://llama.developer.meta.com/docs/api-keys/
LLAMA_KEY = os.getenv("LLAMA_KEY")  # Fill in your Llama API key, Used for LlamaParse()
LLAMA_MODEL = "meta-llama/llama-guard-4-12b"

# Mem0
# see: https://mem0.ai
MEM0_API_KEY = os.getenv("MEM0_API_KEY")  # Fill in your Mem0 API key

# OpenAI
# see: https://openai.com/api/
# see: https://olympus.mygreatlearning.com/courses/129359/modules/items/7809007?pb_id=18908
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # Fill in the OpenAI API base URL (e.g., "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Fill in your OpenAI API Token (from My Great Learning)
OPENAI_EMB_MODEL = "text-embedding-3-small"  # embedding models "text-embedding-ada-002", "text-embedding-3-large"
OPENAI_MODEL = "gpt-4o-mini"  # Fill in the OpenAI model name (e.g., "gpt-4o-mini")

# --- Environment Keys ---
# Note: This line is for WRITING (or modifying) a variable within the Python process's environment.
# Set the cleaned value back into the environment for libraries like LangChain to find
os.environ["HF_TOKEN"] = HF_TOKEN.strip()
os.environ["GROQ_API_KEY"] = GROQ_API_KEY.strip()
os.environ["LLAMA_KEY"] = LLAMA_KEY.strip()
os.environ["MEM0_API_KEY"] = MEM0_API_KEY.strip()
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE.strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY.strip()
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
# --- Environment Keys ---

# --- CONSTANTS ---
EVAL_THRESHOLD = 0.8
EXIT_CMD = "exit"
MILLI_IN_SECS = 1000
RETRIEVAL_LIMIT = 5
SECS_IN_MIN = 60 # secs in min
VECTOR_RESULT_CNT = 3

# Define the Google Drive and other directory paths
COLLECTION_NAME = "nutritional"
DOCUMENT_DIR = "Nutritional Medical Reference"
DOCUMENT_ZIP = "Nutritional_Medical_Reference.zip" # Zip file name

# Prompt variables
ROLE = "Nutrition Disorder Specialist"
TITLE = "SMART NUTRITION DISORDER SPECIALIST BOT"

# Define prompt messages and queries
#
# --- Llama Guard 3 8B for S14 Code Interpreter Abuse ---
# see: https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3
#
# S1:  Violent Crimes.
# S2:  Non-Violent Crimes.
# S3:  Sex Crimes.
# S4:  Child Exploitation.
# S5:  Defamation.
# S6:  Specialized Advice.
# S7:  Privacy.
# S8:  Intellectual Property.
# S9:  Indiscriminate Weapons.
# S10: Hate.
# S11: Self-Harm.
# S12: Sexual Content.
# S13: Elections.
# S14: Code Interpreter Abuse
#
# * We will permit codes S6, S7, S8, and S13 for this Nutrition Disorder Specialist bot. *
#
LLAMA_UNSAFE_CODES = ["S1", "S2", "S3", "S4", "S5", "S9", "S10", "S11", "S12"]

# --- HELPER FUNCTIONS
def show_datetime() -> str:
    now_utc = datetime.now(UTC)

    return now_utc.strftime("%b %d %Y %I:%M:%S %p %Z")

def start_timer() -> float:
    return time.time()


def get_time(start_time_int: float) -> str:
    diff = abs(time.time() - start_time_int)
    _, remainder = divmod(diff, (SECS_IN_MIN*SECS_IN_MIN))
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)
    ms = fractional_seconds * MILLI_IN_SECS

    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"


def show_timer(start_time_int: float) -> None:
    print(f"\nRun Time: {get_time(start_time_int)}")


def filter_response(resp, index=None) -> str:
    # 1. Check if the response is already a string (raw output)
    if isinstance(resp, str):
        content = resp.strip()

    # 2. Check if the response is a LangChain Message object
    elif hasattr(resp, 'content'):
        content = resp.content.strip()

    # 3. Handle unexpected types
    else:
        print(f'Warning: Unexpected response type for chunk {index}. Type: {type(resp)}')
        return "[]" # Treat unexpected types as an empty response

    if len(content) == 0:
        print(f'No generated hypothetical questions found for chunk {index}.')
        return "[]"

    # The output is wrapped in outer quotes and parentheses, e.g., ("['...']")
    # 2. Check for and remove the outer parentheses and quotes if present
    if content.startswith('(') and content.endswith(')'):

        # Remove the outer parentheses
        content = content.strip()[1:-1].strip()

        # Remove the outer quotes that might remain
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]

    # Important: In the LLM-as-string case, you might get "[]" here.
    # The calling code handles the difference between an empty string and the literal string "[]"
    return content


# Initializes vector retriever and gets vectorized data stored in Chroma
# The `persist directory` is from the root repository path, not the Google Colab path.
def get_retriever(coll_name: str):
    vector_store = Chroma(
        collection_name=coll_name,
        embedding_function=embedding_model,
        persist_directory=f"{coll_name}_db"
    )

    # Create a retriever from the vector store
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": VECTOR_RESULT_CNT}
    )


# Checks if all necessary keys are being used
def check_program_keys() -> bool:
    # Load keys and check if any or missing to kill the script.
    keys_to_check = {
        "HF_TOKEN": HF_TOKEN,
        "GROQ_API_KEY": GROQ_API_KEY,
        "LLAMA_KEY": LLAMA_KEY, # This is the alias for os.getenv("LLAMA_KEY")
        "MEM0_API_KEY": MEM0_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY, # formerly config.json("API_KEY")
        "OPENAI_API_BASE": OPENAI_API_BASE,
    }

    missing_keys = []
    for key_name, key_value in keys_to_check.items():
        error_msg = f"{key_name} value is None!"
        if not key_value: # Checks if the value is None (i.e., not found)
            missing_keys.append(key_name)
        if key_value is None:
            st.error(error_msg)
            print(error_msg)

    error_msg = f"FATAL ERROR: The following secrets are missing... {', '.join(missing_keys)}."
    if missing_keys:
        st.error(error_msg)
        print(error_msg)
        return False

    return True


# Checks if the document directory exists
def check_document_file() -> bool:

    if not os.path.isdir(DOCUMENT_DIR):
        st.warning(f"WARNING: Document directory: `{DOCUMENT_DIR}`  not found!")
        print(f"WARNING: Document directory: `{DOCUMENT_DIR}`  not found!")

        # Check for a zip file
        if not os.path.exists(DOCUMENT_ZIP):
            st.error(f"ERROR: Required zip file `{DOCUMENT_ZIP}` not found either.  Please upload it.")
            print(os.listdir('.'))
            return False

        else:
            st.info(f"Zip file: `{DOCUMENT_ZIP}` found!\nExtracting zip file...")
            print(f"Zip file: `{DOCUMENT_ZIP}` found!\nExtracting zip file...")

            with zipfile.ZipFile(DOCUMENT_ZIP, 'r') as zip_ref:
                zip_ref.extractall(".")

            st.info(f"Zip file: `{DOCUMENT_ZIP}` extracted.")
            print(f"Zip file: `{DOCUMENT_ZIP}` extracted.")
            return True

    else:
        print(f"Document directory found: `{DOCUMENT_DIR}`.")
        return True
# --- End of Helper Functions --- #

# --- INITIALIZE PERSISTENT STATE ---
session_keys_valid = None
session_doc_found = None

if session_keys_valid not in st.session_state:
    st.session_state[session_keys_valid] = session_keys_valid

if session_doc_found not in st.session_state:
    st.session_state[session_doc_found] = session_doc_found

# --- VALIDATE API CREDENTIALS KEYS AND CHECK THE SOURCE FILE
if st.session_state[session_keys_valid] is None:
    is_valid = check_program_keys()
    st.session_state[session_keys_valid] = is_valid

    if not is_valid:
        st.stop()

# --- FIND & REFERENCE DOCUMENT FOR CHUNKING
if st.session_state[session_doc_found] is None:
    doc_found = check_document_file()
    st.session_state[session_doc_found] = doc_found

    if not doc_found:
        st.stop()


# --- Start Program --- #
print("# --- START PROGRAM --- #")

# --- FILTER INPUT WITH LLAMA GUARD
# Initialize the Llama Guard client with the API key
llama_guard_client = Groq(api_key=GROQ_API_KEY)

# Initialize the OpenAI Embeddings
# see: https://docs.langchain.com/oss/python/integrations/text_embedding/openai
embedding_model = OpenAIEmbeddings(
    openai_api_base=OPENAI_API_BASE, # Fill in the endpoint
    openai_api_key=OPENAI_API_KEY,   # Fill in the API key
    model=OPENAI_EMB_MODEL,          # Fill in the model name
    max_retries=8,                   # openai client retries, Added for robustness (was =3)
    request_timeout=60,              # avoid timeouts on backoff
)
# This initializes the OpenAI embeddings model using the specified endpoint, API key, and model name.

# Initialize the Chat OpenAI model
llm = ChatOpenAI(
    base_url=OPENAI_API_BASE,         # Fill in the endpoint
    openai_api_key=OPENAI_API_KEY,  # Fill in the API key
    model=OPENAI_MODEL,               # Fill in the deployment name (e.g., gpt-4o-mini)
    streaming=False,
    max_tokens=None,

    # New additions for robustness and quality:
    temperature=0.0,                 # Set for factual, deterministic output
    max_retries=5,                   # Retry failed calls
    # Timeout after 60 seconds
)
# This initializes the Chat OpenAI model using the provided endpoint, API key, deployment name.

# Set the LLM and embedding model in the LlamaIndex settings.
Settings.llm = llm
Settings.embedding = embedding_model

# Apply the nested async loop to allow async code execution in the notebook
nest_asyncio.apply()

# Initialize LlamaParse with desired settings
parser = LlamaParse(
    result_type="markdown",  # Specify the result format
    skip_diagonal_text=True, # Skip diagonal text in the PDFs
    fast_mode=False,         # Use normal mode for parsing
    num_workers=9,           # Number of workers for parallel processing
    check_interval=10,       # Check interval for processing
    api_key=LLAMA_KEY        # API key for LlamaParse
)

# --- INITIALIZE CHROMA VECTOR STORAGE FOR RETRIEVING DOCUMENTS
# Retrieve `nutritional` database created from Google Colab
collection_name = "nutritional"
retriever = get_retriever("nutritional")

# --- DEFINE AGENT STATE
class AgentState(TypedDict):
    query: str  # The current user query
    expanded_query: str  # The expanded version of the user query
    context: List[Dict[str, Any]]  # Retrieved documents (content and metadata)
    response: str  # The generated response to the user query
    precision_score: float  # The precision score of the response
    groundedness_score: float  # The groundedness score of the response
    groundedness_loop_count: int  # Counter for groundedness refinement loops
    precision_loop_count: int  # Counter for precision refinement loops
    feedback: str
    query_feedback: str
    groundedness_check: bool
    loop_max_iter: int
    ROLE: str

# AI AGENT HELPER QUERIES

# Function to filter user input with Llama Guard
def filter_input_with_llama_guard(user_input_str: str, model=LLAMA_MODEL) -> str:
    """
    Filters user input using Llama Guard to ensure it is safe.
    Whitelist "UNSAFE" codes: S6, S7, S8, S13 so that you can handle the customer query.

    Parameters:
    - user_input: The input provided by the user.
    - model: The Llama Guard model to be used for filtering (default is "meta-llama/llama-guard-4-12b").

    Returns:
    - The filtered and safe input.
    """

    try:
        # Create a request to Llama Guard to filter the user input
        llama_response = llama_guard_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": user_input_str
            }],
            model=model,
        )

        # Return the filtered input
        result = llama_response.choices[0].message.content.strip()
        print(f"Guard result: {result}")
        if "unsafe" in result:
            if any(code.strip() in LLAMA_UNSAFE_CODES for code in result.replace("unsafe ", "").strip().split(",")):
                return "BYPASS_SAFE"
            else:
                return "UNSAFE"
        else:
            return "SAFE"

    except Exception as e:
        print(f"Error with Llama Guard: {e}")
        return ""


# --- EXPAND QUERY
def expand_query(state: AgentState) -> AgentState:
    """
    Expands the user query to improve retrieval of nutrition-disorder-related information using few-shot prompting.

    Args:
        state (Dict): The current state of the workflow, containing the user query.

    Returns:
        Dict: The updated state with the expanded query.
    """

    print("\n-------- expand_query ---------")

    original_query = state['query']
    query_feedback = state.get('query_feedback') # Gets feedback if present

    # --- Start with the ROBUST V1 Prompt ---
    system_message = f"""
    You are an expert AI {ROLE} specializing in nutritional disorders and academic literature search.

    Your task is to rewrite the user's query into a single detailed, precise, and technical search query optimized for retrieving relevant academic research papers on nutrition disorders.

    - Incorporate key domain-specific terms, synonyms, and related clinical terminology.
    - Expand abbreviations and clarify ambiguous terms with terminology common in scientific literature.
    - Keep the core clinical intent and meaning intact.

    - **CRITICAL EXCEPTION (V1 INTEGRATION):** If the user's query is **conversational**, **procedural**, or **meta-data related** (e.g., "What can I ask?", "Who are you?"), **DO NOT** expand it. **Return the original user query exactly as provided.**

    - Format the output as a concise query string suitable for academic database search engines.
    - Your output MUST be only the rewritten query string and nothing else.
    """

    if query_feedback:
        print("--- Using feedback to refine query ---")

        system_message += f"""

        You have already generated a query that was not precise enough. Use the following SUGGESTIONS to create a NEW, improved query.

        SUGGESTIONS:
        {query_feedback}
        """

    # Create the final prompt template
    expand_prompt = CoreChatPromptTemplate.from_messages([
        ("system", system_message.strip()),
        ("user", "Original User Query: {query}")
    ])

    chain = expand_prompt | llm | StrOutputParser()

    # Invoke the chain
    state['expanded_query'] = chain.invoke({
        "query": original_query,
        # Note: Feedback is injected via the system_message,
        "ROLE": state['ROLE'],
    })

    # Clear the feedback for the next node
    state['query_feedback'] = ""

    return state


# --- RETRIEVE CONTEXT
def retrieve_context(state: AgentState) -> AgentState:
    """
    Retrieves context from the vector store using the expanded or original query.

    Args:
        state (Dict): The current state of the workflow, containing the query and expanded query.

    Returns:
        Dict: The updated state with the retrieved context.
    """

    query = state['expanded_query']

    print("\n--- retrieve_context ---")
    print("Query used for retrieval:", query)  # Debugging: Print the query

    # Retrieve documents from the vector store
    retrieved_docs = retriever.invoke(query)

    print("Retrieved documents:", retrieved_docs)  # Debugging: Print the raw docs object

    # Extract both page_content and metadata from each document
    state['context'] = [
        {
            "content": doc.page_content,  # The actual content of the document
            "metadata": doc.metadata  # The metadata (e.g., source, page number, etc.)
        }
        for doc in retrieved_docs
    ]

    print("Extracted context with metadata:", state['context'])  # Debugging: Print the extracted context

    return state


# --- CRAFT RESPONSE
def craft_response(state: Dict) -> Dict:
    """
    Generates a response using the retrieved context, focusing on nutrition disorders.

    Args:
        state (Dict): The current state of the workflow, containing the query and retrieved context.

    Returns:
        Dict: The updated state with the generated response.
    """
    print("\n--- craft_response ---")

    system_message = """
    You are an expert AI {ROLE}, specializing in **Nutritional Disorders**. Your sole task is to analyze the provided CONTEXT and synthesize a direct, comprehensive answer to the user's QUERY.

    **STRICT GENERATION RULES:**
    1.  **Groundedness:** Generate the response using **ONLY** the information found in the retrieved CONTEXT. Do not use outside knowledge.
    2.  **Format:** Output the answer as a structured, numbered list of concise, clinically relevant statements.
    3.  **Incorporation:** If the 'FEEDBACK' suggests improvements, use it to refine the response based on the CONTEXT. If there is no FEEDBACK, ignore it.
    4.  **Clinical Detail:** Include **key numeric thresholds**, **dosage recommendations**, and **specific diagnostic criteria** exactly as they appear in the CONTEXT.
    5.  **Citations:** Append the source document/page number to each statement if available in the CONTEXT.

    **INCOMPLETENESS:**
    If the retrieved CONTEXT is insufficient to answer the query, your entire response must be: **"The retrieved context is insufficient to provide a complete and grounded answer. Further clinical follow-up is recommended."**

    **DO NOT** include any commentary, greetings, or introductory/concluding remarks outside of the numbered list.
    """

    response_prompt = CoreChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nContext: {context}\n\nfeedback: {feedback}")
    ])

    chain = response_prompt | llm
    response = chain.invoke({
        "query": state['query'],
        "context": "\n".join([doc["content"] for doc in state['context']]),
        "feedback": state["feedback"], # add feedback to the prompt
        "ROLE": state["ROLE"],
    })

    state['response'] = response

    print("intermediate response: ", response)

    return state


# --- SCORE GROUNDEDNESS
def score_groundedness(state: Dict) -> Dict:
    """
    Checks whether the response is grounded in the retrieved context.

    Args:
        state (Dict): The current state of the workflow, containing the response and context.

    Returns:
        Dict: The updated state with the groundedness score.
    """

    print("\n--- check_groundedness ---")

    system_message = """You are a meticulous AI {ROLE} Quality Analyst and fact-checker. Your sole task is to evaluate how well a given response is supported by a provided context.
    Calculate a score from 0.0 to 1.0 that represents the fraction of claims in the response that are directly and verifiably supported by the context.
    - A score of 1.0 means every claim in the response is fully supported by the context.
    - A score of 0.0 means no claims in the response are supported by the context.

    **Your output MUST be the numerical score as a single float. Do not output any other text, explanation, or markdown.**
    """

    groundedness_prompt = CoreChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Context: {context}\nResponse: {response}\n\nGroundedness score:")
    ])

    chain = groundedness_prompt | llm | StrOutputParser()
    groundedness_score = float(chain.invoke({
        "context": "\n".join([doc["content"] for doc in state['context']]),
        "response": state['response'],
        "ROLE": state["ROLE"],
    }))


    state['groundedness_loop_count'] += 1

    print("groundedness_score: ", groundedness_score)
    print("######## Groundedness Incremented ##########")

    state['groundedness_score'] = groundedness_score

    return state


# --- CHECK PRECISION
def check_precision(state: Dict) -> Dict:
    """
    Checks whether the response precisely addresses the user’s query.

    Args:
        state (Dict): The current state of the workflow, containing the query and response.

    Returns:
        Dict: The updated state with the precision score.
    """

    print("\n--- check_precision ---")

    system_message = """
    As an AI {ROLE} evaluate whether the response precisely addresses the user's query.
    Evaluate, assign and return the precision score for the response.  Your evaluation is based solely on the relationship between the response and the query. Do not consider anything else.

    The score is from 0.0 (least) to 1.0 (best).
    - A score of 1.0 means the response is precise, on-topic and accurately addressed by the query.
    - A score of 0.0 means the response is ambiguous, off-topic, or inaccurate and does not accurately address the query.

    **Only output the numerical score as a float and nothing else!**
    """

    precision_prompt = CoreChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nResponse: {response}\n\nPrecision score:")
    ])

    chain = precision_prompt | llm | StrOutputParser()
    precision_score = float(chain.invoke({
        "query": state['query'],
        "response": state['response'],
        "ROLE": state["ROLE"],
    }))

    state['precision_score'] = precision_score
    state['precision_loop_count'] += 1

    print("precision_score:", precision_score)
    print("######## Precision Incremented ##########")

    return state


# --- REFINE RESPONSE
def refine_response(state: Dict) -> Dict:
    """
    Suggests improvements for the generated response.

    Args:
        state (Dict): The current state of the workflow, containing the query and response.

    Returns:
        Dict: The updated state with response refinement suggestions.
    """

    print("\n--- refine_response ---")

    system_message = """
    You are an AI {ROLE} Quality Analyst and Critic. Your sole task is to provide constructive feedback on a given response based on the user's original query.
    Your feedback should identify potential gaps, ambiguities, or missing details and suggest specific improvements to enhance the response's accuracy and completeness.

    - Use bullet points to structure your suggestions.
    - Do NOT rewrite the full response. Only provide a list of suggestions for improvement.
    - Your output must be only the bulleted list of suggestions. Do not include a preamble like "Here are my suggestions:".
    """

    refine_response_prompt = CoreChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nResponse: {response}\n\n"
                 "What improvements can be made to enhance accuracy and completeness?")
    ])

    chain = refine_response_prompt | llm | StrOutputParser()

    # Store response suggestions in a structured format
    feedback = f"Previous Response: {state['response']}\nSuggestions: {chain.invoke({'query': state['query'], 'response': state['response'], 'ROLE': state['ROLE']})}"

    print("feedback: ", feedback)
    print(f"State: {state}")

    state['feedback'] = feedback

    return state


# --- REFINE QUERY
def refine_query(state: Dict) -> Dict:
    """
    Suggests improvements for the expanded query, returning them in a structured JSON format.

    Args:
        state (Dict): The current state of the workflow, containing the query and expanded query.

    Returns:
        Dict: The updated state with JSON-formatted query refinement suggestions.
    """

    print("\n--- refine_query ---")

    # Define a Pydantic model that matches the desired JSON structure.
    # This is the correct way to provide a schema to JsonOutputParser.
    class QuerySuggestions(BaseModel):
        missing_keywords: List[str]
        scope_refinements: List[str]
        term_clarifications: List[str]

    # This prompt forces the JSON structure and ensures high-quality clinical input
    system_message = f"""
    You are an AI Search Query Analyst specializing in clinical nutrition literature.
    Your sole task is to provide constructive feedback on the provided expanded query to enhance its search precision for academic databases.

    - Do NOT rewrite the query.
    - Analyze the Expanded Query against the Original Query and suggest improvements only in the required JSON format.
    - If a category has no suggestions, return an empty list for that key.

    - Your output MUST be a JSON object that strictly adheres to the format defined by the tool.
    """

    # Use the LangChain JsonOutputParser for reliable structured output
    json_parser = JsonOutputParser(pydantic_object=QuerySuggestions)

    refine_query_prompt = CoreChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Original Query: {query}\nExpanded Query to Critique: {expanded_query}\n\nProvide your JSON suggestions:")
    ])

    chain = refine_query_prompt | llm | json_parser

    # Invoke the chain to get structured suggestions
    suggestions = chain.invoke({
        "query": state['query'],
        "expanded_query": state['expanded_query'],
        "ROLE": state["ROLE"],
    })

    # Store the JSON object as a string in the state for the next node to consume
    suggestions_str = json.dumps(suggestions, indent=2)

    state['query_feedback'] = suggestions_str

    print(f"Query Feedback Generated (JSON):\n{suggestions_str}")

    return state


# --- HAS MAX ITERATIONS REACHED?
# Checks if the maximum number of iterations has been reached
# Note: This method must be before should_* methods.
def has_max_iterations_reached(state: Dict, var: str) -> bool:
    return state[var] >= state["loop_max_iter"]


# --- CHECK GROUNDEDNESS
def should_continue_groundedness(state):

    """Decides if groundedness is enough or needs improvement."""

    print("--- should_continue_groundedness ---")
    print("groundedness loop count: ", state['groundedness_loop_count'])

    if state["groundedness_score"] >= EVAL_THRESHOLD:  # Threshold for groundedness
        print("Moving to precision")

        return "check_precision"

    else:
        if has_max_iterations_reached(state, "groundedness_loop_count"):
            return "max_iterations_reached"
        else:
            print(f"--- Groundedness Score Threshold Not met. Refining Response -----")

            return "refine_response"


# --- CHECK PRECISION
def should_continue_precision(state: Dict) -> str:

    """Decides if precision is enough or needs improvement."""

    print("--- should_continue_precision ---")
    print("precision loop count: ", state['precision_loop_count'])

    if state["precision_score"] >= EVAL_THRESHOLD:  # Threshold for precision
        return "pass"  # Complete the workflow

    else:
        if has_max_iterations_reached(state, "precision_loop_count"):  # Maximum allowed loops
            return "max_iterations_reached"
        else:
            print(f"--- Precision Score Threshold Not met. Refining Query ---")  # Debugging

            return "refine_query"  # Refine the query


# --- MAX ITERATIONS REACHED
def max_iterations_reached(state: AgentState) -> AgentState:
    """Handles the case where max iterations are reached."""
    state['response'] = "We need more context to provide an accurate answer."
    return state


# --- CREATE WORKFLOW
# Used for LineGraph (library of Agentic RAG), a workflow is modeled as a StateGraph, which is simply a state machine (like a complex flowchart).
def create_workflow() -> StateGraph:

    """Creates the updated workflow for the AI nutrition agent."""
    workflow = StateGraph(AgentState)

    # Add processing nodes
    workflow.add_node("expand_query", expand_query)                     # Step 1: Expand user query.
    workflow.add_node("retrieve_context", retrieve_context)             # Step 2: Retrieve relevant documents.
    workflow.add_node("craft_response", craft_response)                 # Step 3: Generate a response based on retrieved data.
    workflow.add_node("score_groundedness", score_groundedness)         # Step 4: Evaluate response grounding.
    workflow.add_node("refine_response", refine_response)               # Step 5: Improve response if it's weakly grounded.
    workflow.add_node("check_precision", check_precision)               # Step 6: Evaluate response precision.
    workflow.add_node("refine_query", refine_query)                     # Step 7: Improve query if response lacks precision.
    workflow.add_node("max_iterations_reached", max_iterations_reached) # Step 8: Handle max iterations.

    # Define the entry point where to start
    workflow.set_entry_point("expand_query")

    # Main flow edges
    workflow.add_edge("expand_query", "retrieve_context")
    workflow.add_edge("retrieve_context", "craft_response")
    workflow.add_edge("craft_response", "score_groundedness")

    # Conditional edges based on groundedness check
    workflow.add_conditional_edges(
        "score_groundedness",
        should_continue_groundedness,  # Use the conditional function
        {
            "check_precision": "check_precision",              # If well-grounded, proceed to precision check.
            "refine_response": "refine_response",              # If not, refine the response.
            "max_iterations_reached": "max_iterations_reached" # If max loops reached, exit.
        }
    )

    workflow.add_edge("refine_response", "craft_response")  # Refined responses are reprocessed.

    # Conditional edges based on precision check
    workflow.add_conditional_edges(
        "check_precision",
        should_continue_precision,  # Use the conditional function
        {
            "pass": END,                     # If precise, complete the workflow.
            "refine_query": "refine_query",  # If imprecise, refine the query.
            "max_iterations_reached": "max_iterations_reached"    # If max loops reached, exit.
        }
    )

    workflow.add_edge("refine_query", "expand_query") # Refined queries go through expansion again.
    workflow.add_edge("max_iterations_reached", END)

    return workflow


# --- VISUALIZE WORKFLOW
WORKFLOW_APP = create_workflow().compile()


# --- INITIALIZE AGENTIC RETRIEVAL AUGMENTED GENERATION (RAG)
@tool
def agentic_rag(query: str):
    """
    Runs the RAG-based agent with conversation history for context-aware responses.

    Args:
        query (str): The current user query.

    Returns:
        Dict[str, Any]: The updated state with the generated response and conversation history.
    """
    # Initialize state with necessary parameters
    inputs = {
        "query": query,
        "expanded_query": "",
        "context": [],
        "response": "",
        "precision_score": 0.0,
        "groundedness_score": 0.0,
        "groundedness_loop_count": 0,
        "precision_loop_count": 0,
        "feedback": "",
        "query_feedback": "",
        "loop_max_iter": 4,
        "ROLE": ROLE
    }

    return WORKFLOW_APP.invoke(inputs)


# --- DECLARE NUTRITION BOT
class NutritionBot:
    def __init__(self):
        """
        # see: https://olympus.mygreatlearning.com/courses/129359/modules/items/7899896?pb_id=18908

        Initialize the NutritionBot class, setting up memory, the LLM client, tools, and the agent executor.
        """

        # Initialize a memory client to store and retrieve customer interactions
        self.memory = MemoryClient(api_key=MEM0_API_KEY)  # Complete the code to define the memory client API key

        # Initialize the OpenAI client using the provided credentials
        self.client = ChatOpenAI(
            model_name=OPENAI_MODEL,  # Specify the model to use (e.g., a GPT-4 optimized version)
            openai_api_key=OPENAI_API_KEY,  # API key for authentication
            base_url = OPENAI_API_BASE,
            temperature=0  # Controls randomness in responses; 0 ensures deterministic results
        )

        # Define tools available to the chatbot, such as web search
        tools = [agentic_rag]

        # Define the system prompt to set the behavior of the chatbot
        system_prompt = """You are a caring and knowledgeable Medical Support Agent, specializing in nutrition disorder-related guidance. Your goal is to provide accurate, empathetic, and tailored nutritional recommendations while ensuring a seamless customer experience.
                          Guidelines for Interaction:
                          Maintain a polite, professional, and reassuring tone.
                          Show genuine empathy for customer concerns and health challenges.
                          Reference past interactions to provide personalized and consistent advice.
                          Engage with the customer by asking about their food preferences, dietary restrictions, and lifestyle before offering recommendations.
                          Ensure consistent and accurate information across conversations.
                          If any detail is unclear or missing, proactively ask for clarification.
                          Always use the agentic_rag tool to retrieve up-to-date and evidence-based nutrition insights.
                          Keep track of ongoing issues and follow-ups to ensure continuity in support.
                          Your primary goal is to help customers make informed nutrition decisions that align with their health conditions and personal preferences.
        """

        # Build the prompt template for the agent
        prompt = CoreChatPromptTemplate.from_messages([
            ("system", system_prompt),             # System instructions
            ("human", "{input}"),                  # Placeholder for human input
            ("placeholder", "{agent_scratchpad}")  # Placeholder for intermediate reasoning steps
        ])

        # Create an agent capable of interacting with tools and executing tasks
        agent = create_tool_calling_agent(self.client, tools, prompt)

        # Wrap the agent in an executor to manage tool interactions and execution flow
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


    def store_customer_interaction(self, user_id: str, message: str, response: str, metadata: Dict = None):
        """
        Store customer interaction in memory for future reference.

        Args:
            user_id (str): Unique identifier for the customer.
            message (str): Customer's query or message.
            response (str): Chatbot's response.
            metadata (Dict, optional): Additional metadata for the interaction.
        """
        if metadata is None:
            metadata = {}

        # Add a timestamp to the metadata for tracking purposes
        metadata["timestamp"] = datetime.now().isoformat()

        # Format the conversation for storage
        conversation = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]

        # Store the interaction in the memory client
        self.memory.add(
            conversation,
            user_id=user_id,
            output_format="v1.1",
            metadata=metadata
        )


    def get_relevant_history(self, user_id: str, query: str) -> List[Dict]:
        """
        Retrieve past interactions relevant to the current query.

        Args:
            user_id (str): Unique identifier for the customer.
            query (str): The customer's current query.

        Returns:
            List[Dict]: A list of relevant past interactions.
        """
        return self.memory.search(
            query=query,  # Search for interactions related to the query
            user_id=user_id,  # Restrict search to the specific user
            limit=RETRIEVAL_LIMIT  # Complete the code to define the limit for retrieved interactions
        )


    def handle_customer_query(self, user_id: str, query: str) -> str:
        """
        Process a customer's query and provide a response, taking into account past interactions.

        Args:
            user_id (str): Unique identifier for the customer.
            query (str): Customer's query.

        Returns:
            str: Chatbot's response.
        """

        # Retrieve relevant past interactions for context
        relevant_history = self.get_relevant_history(user_id, query)

        # Build a context string from the relevant history
        context = "Previous relevant interactions:\n"
        for memory in relevant_history:
            context += f"Customer: {memory['memory']}\n"  # Customer's past messages
            context += f"Support: {memory['memory']}\n"  # Chatbot's past responses
            context += "---\n"

        # Print context for debugging purposes
        print("Context: ", context)

        # Prepare a prompt combining past context and the current query
        prompt = f"""
        Context:
        {context}

        Current customer query: {query}

        Provide a helpful response that takes into account any relevant past interactions.
        """

        # Generate a response using the agent
        response = self.agent_executor.invoke({"input": prompt})

        # Store the current interaction for future reference
        self.store_customer_interaction(
            user_id=user_id,
            message=query,
            response=response["output"],
            metadata={"type": "support_query"}
        )

        # Return the chatbot's response
        return response['output']


# Cache ChatBot Instance
@st.cache_resource
def get_chatbot_instance():
    """Initializes and caches the NutritionBot instance."""
    print("--- Initializing NutritionBot ---")
    return NutritionBot()


# --- DECLARE NUTRITION DISORDER AGENT
def nutrition_disorder_streamlit():
    """
    A Streamlit-based UI for the Nutrition Disorder Specialist Agent.
    """
    st.title(f"{TITLE}")
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

# Ensure you have the necessary classes and functions defined in your main script
if __name__ == "__main__":
    # --- RUN THE AI AGENT --- #
    start_time = start_timer()
    nutrition_disorder_streamlit()
    show_timer(start_time)

# --- END OF PROGRAM --- #
