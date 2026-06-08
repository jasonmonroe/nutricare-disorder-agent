# notebooks/nutricare_disorder_agent.py

!pip install numpy==1.26.4

# Installing the required libraries
!pip install openai==1.55.3 \
            langchain==0.2.7 \
            langchain-community==0.2.7\
             langchain-huggingface==0.0.3 \
             langchain-experimental==0.0.62 \
             langchain-openai==0.1.14 \
             chromadb==0.5.3 \
             sentence-transformers==3.0.1 \
             python-dotenv==1.0.1 \
             lark==1.1.9 \
             llama-index-core \
             llama-parse==0.5.11 \
             llama-index-readers-file \
             llama-index-llms-langchain \
             langgraph \
             groq \
             mem0ai

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import chromadb
import numpy as np
import json

np.float_ = np.float64

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.text_splitter import (
    CharacterTextSplitter,  # Splitting text by characters
    RecursiveCharacterTextSplitter  # Recursive splitting of text by characters
)

# Import LlamaParse for parsing documents
from llama_parse import LlamaParse
from llama_index.core import Settings, SimpleDirectoryReader

# Section 1 Imports (moved)
from google.colab import drive, userdata
from sqlalchemy.testing.plugin.plugin_base import requirements

from datetime import datetime, UTC
import textwrap
import time
import random
import hashlib
import logging
import re

# Import the necessary library for async operations in notebooks
import nest_asyncio
import shutil

# Unzipping the nutritoin medical reference documents into the Nutritional Medical Reference folder
!unzip Nutritional_Medical_Reference.zip

# Define a function to read a JSON config file and return its contents as a dictionary.
def read_config(config_file):
  """Reads a JSON config file and returns a dictionary."""
  with open(config_file, 'r') as f:
    return json.load(f)

# Copy and paste the path of the config file uploaded in Colab
config = read_config("config_GANLP.json")
config_emb = read_config("config2_emb_tested.json")

# --- AI Vendor Credentials ---

# Hugging Face
# see: https://hugginface.co
# see: Model -> https://huggingface.co/jasonmonroe/smart-nutri-disorder-specialist-model
# see: Space -> https://huggingface.co/jasonmonroe/smart-nutri-disorder-specialist-bot
# Fill in the Hugging Face repository ID (e.g., "google/flan-t5-xxl")
HF_REPO_ID = "jasonmonroe/smart-nutri-disorder-specialist-bot"
HF_TOKEN = userdata.get("HF_TOKEN")

# Groq
# see: https://www.groq.com/
GROQ_API_KEY = userdata.get("GROQ_API_KEY")

# Llama
# see: https://llama.developer.meta.com/docs/api-keys/
LLAMA_KEY = config_emb.get("LLAMA_KEY")  # Used for LlamaParse()
LLAMA_MODEL = "meta-llama/llama-guard-4-12b"

# Mem0
# see: https://mem0.ai
MEM0_API_KEY = userdata.get("MEM0_API_KEY")  # Fill in your Mem0 API key

# OpenAI
# see: https://openai.com/api/
# see: https://olympus.mygreatlearning.com/courses/129359/modules/items/7809007?pb_id=18908
OPENAI_API_BASE = config.get("OPENAI_API_BASE")  # Fill in the OpenAI API base URL (e.g., "https://api.openai.com/v1")
OPENAI_API_KEY = config.get("API_KEY")  # Fill in your OpenAI API Token (from My Great Learning)
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # embedding models "text-embedding-3-small", "text-embedding-3-large"
OPENAI_MODEL = "gpt-4o-mini"  # Fill in the OpenAI model name (e.g., "gpt-4o-mini")

# --- CONSTANTS --- #

# Batch sizes (per batch) for processing documents and text chunks
CHUNK_DOC_BATCH_SIZE = 100
CHUNK_TEXT_BATCH_SIZE = 50
EMPTY_RESP = "[]" # Empty response
EVAL_THRESHOLD = 0.8
EXIT_CMD = "exit"
RETRIEVAL_LIMIT = 5
SECS_IN_MIN = 60 # secs in min
SEMANTIC_THRESH_LIMIT = random.randint(80, 85)
VECTOR_RESULT_CNT = 3

# Define the Google Drive and other directory paths
GOOGLE_DRIVE_PATH = "/content/drive/"
DOCUMENT_DIR = "Nutritional Medical Reference"
DOCUMENT_ZIP = "Nutritional_Medical_Reference.zip"
HOME_PATH = "My Drive/Colab Notebooks/GA-NLP/project-final/"
VECTORS_DIR= "vectorstore/"

# Prompt variables
PROMPT_INSTR = """
    Important:
    Generate only a Python list of relevant questions (e.g., ['Question 1', 'Question 2']).
    *Do NOT mention or output anything before or after the list, including commentary, markdown blocks, or extra punctuation.
    If the content cannot answer any question(s), your output MUST be the empty Python list: [].
    """

ROLE = "Nutrition Disorder Specialist"
TITLE = "SMART NUTRITION DISORDER SPECIALIST BOT"

# ----- Helper Functions ----- #

def show_datetime() -> str:
    now_utc = datetime.now(UTC)
    return now_utc.strftime("%b %d %Y %I:%M:%S %p %Z")


def start_timer() -> float:
    return time.time()


def get_time(start_time_int: float) -> str:
    diff = abs(time.time() - start_time_int)
    hours, remainder = divmod(diff, SECS_IN_MIN*SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * 1000
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


def handle_rate_limit_error(e, subject: str, current_sleep_time: int) -> tuple[int, bool]:
    """
    Checks for a Rate Limit Error (429), calculates a new sleep time,
    and returns the new sleep time and a flag indicating the hit.
    """
    print(f"{i}) Exception invoking a response for {subject}! Error: {e}")

    rate_limit_hit = False
    new_sleep_time = current_sleep_time

    if "Error code: 429" in str(e):
        # Increase sleep time by 15%
        rate_limit_hit = True
        new_sleep_time = current_sleep_time + round(current_sleep_time * 0.15)
        print(f"FATAL: Rate limit hit. Updating sleep time from {current_sleep_time} to {new_sleep_time} seconds...")
        if new_sleep_time > SECS_IN_MIN:
            new_sleep_time = SECS_IN_MIN

    return new_sleep_time, rate_limit_hit


# Generate a unique ID for document content
def gen_id(source: str, page_no: int, content: str) -> str:
    id_str = f"{source}|page{page_no}|{content[:64]}"

    return str(hashlib.sha256(id_str.encode('utf-8')).hexdigest())


# Show a random document sample
def show_sample(samp_docs, samp_title="") -> None:
    doc_cnt = len(samp_docs)
    index = random.randint(0, doc_cnt-1)
    print(f"Index = {index}, Count = {doc_cnt}")

    # Check if the index is within bounds
    if 0 <= index < doc_cnt:
        print("ID:", samp_docs[index].id, "\n")
        print("Metadata:")
        print(json.dumps(samp_docs[index].metadata, indent=4), "\n")
        print(f"{samp_title}:\n", samp_docs[index].page_content)

    else:
        print(f"\nIndex {index} is out of range for the list with length {doc_cnt}.")


# Format persist directory
def format_dir(path: str) -> str:
    return f"./{VECTORS_DIR}/{path}_db"


# Creates and returns a Document object with metadata
# see: https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html
def create_document(content: str, metadata: dict) -> Document:
    metadata["doc_id"] = gen_id(metadata["source"], metadata["page"], content)

    if "type" not in metadata:
        metadata["type"] = "Document"

    return Document(
        id=metadata["doc_id"],
        page_content=content,
        metadata=metadata,
        type=metadata["type"],
    )


# Sets a flag to show debugging and query logs for agent.
# Used for Google Colab Only
def set_agent_logs() -> bool:

    input_str = input("Show query logs? (Y/N) ").strip().upper()

    if input_str == "Y":
        show_logs = True
    else:
        show_logs = False

    return show_logs

# Adds a list of documents to a specified Chroma collection in batches.
def add_docs_to_vectorstore(
    documents: List[Document],
    collection_name: str,
    persist_dir: str,
    batch_size: int = CHUNK_TEXT_BATCH_SIZE
    ):

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=format_dir(persist_dir)
    )

    for i in range(0, len(documents), batch_size):
        batch = documents[i: i + batch_size]
        vectorstore.add_documents(batch)

    print(f"Stored {len(documents)} documents in Chroma collection: {collection_name} at (persist_dir: {persist_dir}))")
    return vectorstore

# Initialize the OpenAI embedding function for Chroma
embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_base=OPENAI_API_BASE,   # Fill in the API base URL
    api_key=OPENAI_API_KEY,   # Fill in the API key
    model_name=OPENAI_EMBEDDING_MODEL, # Fill in the model name
)
# This initializes the OpenAI embedding function for the Chroma vectorstore, using the provided endpoint and API key.

# Initialize the OpenAI Embeddings
# see: https://docs.langchain.com/oss/python/integrations/text_embedding/openai
embedding_model = OpenAIEmbeddings(
    openai_api_base=OPENAI_API_BASE,  # Fill in the endpoint
    openai_api_key=OPENAI_API_KEY,    # Fill in the API key
    model=OPENAI_EMBEDDING_MODEL,           # Fill in the model name
    max_retries=3,                    # Added for robustness
)
# This initializes the OpenAI embeddings model using the specified endpoint, API key, and model name.

# Initialize the Chat OpenAI model
llm = ChatOpenAI(
    base_url=OPENAI_API_BASE,         # Fill in the endpoint
    openai_api_key=OPENAI_API_KEY,    # Fill in the API key
    model=OPENAI_MODEL,               # Fill in the deployment name (e.g., gpt-4o-mini)
    streaming=False,
    max_tokens=None,

    # New additions for robustness and quality:
    temperature=0.0,                 # Set for factual, deterministic output
    max_retries=5,                   # Retry failed calls
    request_timeout=SECS_IN_MIN,     # Timeout after 60 seconds
)
# This initializes the Chat OpenAI model using the provided endpoint, API key, deployment name.

# set the LLM and embedding model in the LlamaIndex settings.
Settings.llm = llm
Settings.embedding = embedding_model

# Apply the nested async loop to allow async code execution in the notebook
nest_asyncio.apply()

# Initialize LlamaParse with desired settings
parser = LlamaParse(
    result_type="markdown",  # Specify the result format
    skip_diagonal_text=True,  # Skip diagonal text in the PDFs
    fast_mode=False,  # Use normal mode for parsing
    num_workers=9,  # Number of workers for parallel processing
    check_interval=10,  # Check interval for processing
    api_key=LLAMA_KEY  # API key for LlamaParse
)

start_time = start_timer()

# List to store parsed JSON objects
json_objs = []

# Define the folder containing the documents
folder_path = DOCUMENT_DIR

# Iterate through PDFs in the folder and parse content
for pdf in os.listdir(folder_path):
    if pdf.endswith(".pdf"):
        pdf_path = os.path.join(folder_path, pdf)
        json_objs.extend(parser.get_json_result(pdf_path))

show_timer(start_time)

# Pretty print JSON object for inspection
print(json.dumps(json_objs[0], indent=4))

# Revised Cell to properly get text from document pages: {page_texts}.
# Initialize dictionaries to store page texts and tables
page_texts, tables = {}, {}

# Extract tables and adjacent text from the parsed JSON objects
for obj in json_objs:
    json_list = obj['pages']
    name = obj["file_path"].split("/")[-1] # Extract the file name

    page_texts[name] = {}
    tables[name] = {}

    for json_item in json_list:
        page_number = json_item["page"]

        # 1. Check for and store table data from the 'items' array
        table_ref_string = None
        table_rows = None

        for component in json_item['items']:
            if component['type'] == 'table': # Check if the component is a table
                table_rows = component['rows']
                # The text field in the item component often holds the table title/reference
                # This is the string we need to remove from the overall page text.
                # We'll rely on the main page 'text' field for removal.

                # Check the overall page text for a table reference string
                # We will search for a pattern like '[Table 1-1. ...]'

                # This pattern finds any text inside brackets that looks like a table reference
                # We look at the 'text' field of the component if it exists, otherwise rely on manual inspection.

                # This pattern searches the page text for the exact table title
                try:
                    # Find the table title text just before the table component starts
                    # We can use the text from the previous item or just the general table placeholder text if available
                    # Based on your example, the text is '[Table 1-1. Glycemic Index of Some Foods]'
                    table_ref_string = next(
                        (
                            item['value'] for item in json_item['items']
                            if item['type'] == 'text' and 'Table' in item['value']
                        ), None
                    )

                    # A more reliable way based on the overall page text:
                    table_ref_match = re.search(r'(\[Table\s*[\d\-\.]+\.\s*[^\]]+\])', json_item['text'])
                    if table_ref_match:
                        table_ref_string = table_ref_match.group(1)

                except Exception as e:
                    print(f"No table ref string. Error: {e}")
                    table_ref_string = None # Failed to find the specific table reference text

                # Store the table data
                tables[name][page_number] = table_rows
                break # Assuming max one table per page for context extraction

        # 2. Extract the clean adjacent text context
        page_content_full = json_item['text']
        page_content_clean = page_content_full

        # Remove the table reference string if found
        if table_rows and table_ref_string:
            page_content_clean = page_content_full.replace(table_ref_string, "").strip()

        # If no explicit reference string was found, we still need to strip the content
        # For simplicity, if a table exists, we remove common boilerplate text around tables
        elif table_rows:
            # Look for lines that contain the table header/title implicitly
            lines = page_content_full.split('\n')
            clean_lines = []

            for line in lines:
                # Heuristically remove lines that look like table headers or footers
                if 'Table' in line and any(c.isdigit() for c in line):
                    continue

                # Also remove the page numbers/footers
                if line.strip().isdigit() and len(line.strip()) < 4:
                    continue

                clean_lines.append(line)

            page_content_clean = "\n".join(clean_lines).strip()

        # Store the final context text
        page_texts[name][page_number] = page_content_clean

"""
# deprecated - Original Cell
# Initialize dictionaries to store page texts and tables
page_texts, tables = {}, {}

# Extract tables from the parsed JSON objects
for obj in json_objs:
    json_list = obj['pages']
    name = obj["file_path"].split("/")[-1]  # Extract the file name
    page_texts[name] = {}  # Initialize dictionary for page texts
    tables[name] = {}  # Initialize dictionary for tables

    for json_item in json_list:
        for component in json_item['items']:
            if component['type'] == 'table':  # Check if the component is a table
                tables[name][json_item['page']] = component['rows']  # Store table rows
"""

print(json.dumps(page_texts, indent=4))

# Display extracted tables for each PDF
for file_name, file_tables in tables.items():
    print(f"Tables from {file_name}:")
    for page_num, table_rows in file_tables.items():
        print(f"Page {page_num}:")
        for row in table_rows:
            print(f"\t{row}")

# Initialize ChromaDB client
os.environ["CHROMA_SERVER_NO_TELEMETRY"] = "true"
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

chromadb_client = chromadb.EphemeralClient()

semantic_text_splitter = SemanticChunker(
    embedding_model,  # Fill in the embedding model
    breakpoint_threshold_type='percentile',  # Choose the threshold type (e.g., 'percentile')
    breakpoint_threshold_amount=SEMANTIC_THRESH_LIMIT # Set the chunking threshold (e.g., 80, 85)
)
# This initializes the semantic text splitter, controlling how the text is divided into meaningful chunks.

start_time = start_timer()

# Step 1: Define the folder containing the documents
#folder_path = "__________"  # Fill in the folder name (e.g., "Nutritional Medical Reference")

# Step 2: Initialize an empty list to store all semantic chunks
semantic_chunks = []

# Step 3: Initialize the PyPDFDirectoryLoader for the folder
pdf_loader = PyPDFDirectoryLoader(folder_path)  # Use the correct loader (e.g., PyPDFDirectoryLoader)

# Step 4: Load and split PDF documents into chunks using SemanticChunker
chunks = pdf_loader.load_and_split(semantic_text_splitter)  # Call the appropriate function and pass the splitter

# Step 5: Extend the semantic_chunks list with the chunks from this folder
semantic_chunks.extend(chunks)  # Add chunks to the list

# Step 6: Get the total number of chunks
print(f"Total Semantic Chunks Created: {len(semantic_chunks)}")

show_timer(start_time)

# Add IDs to the semantic chunks

#semantic_chunks = [Document(id=i, page_content=d.page_content, metadata=d.metadata) for i, d in enumerate(semantic_chunks)]
semantic_chunks = [create_document(d.page_content, d.metadata) for i, d in enumerate(semantic_chunks)]

# Plot a histogram of the number of characters in each semantic chunk
import matplotlib.pyplot as plt

chunk_lengths = [len(chunk.page_content) for chunk in semantic_chunks]
plt.hist(chunk_lengths, bins=25, edgecolor='black')
plt.xlabel('Chunk Length')
plt.ylabel('Frequency')
plt.title('Number of Characters in Each Semantic Chunk')
plt.show()

# Store the document chunks in Chroma vectorstore in batches
# Initialize the vector store once
semanticstore = Chroma(
    embedding_function=embedding_model,
    collection_name="semantic_chunks",
    persist_directory=format_dir("research")
)

batch_size = CHUNK_DOC_BATCH_SIZE  # Adjust the batch size as needed
for i in range(0, len(semantic_chunks), batch_size):
    batch = semantic_chunks[i : i + batch_size]
    semanticstore.add_documents(batch)

# Used for searching the document for pairing subject for building an effective RAG system.
# This query asks a specific question about vitamin deficiencies and memory impairment.
# Match keywords and map them with embedding models.
SIMILARITY_SEARCH_QUERY = "What nutritional deficiency, such as folate deficiency or that caused by alcoholism, is clinically linked to anemia, and what specific standard diagnostic metric is used for its confirmation?"

user_input = SIMILARITY_SEARCH_QUERY  # Fill in a query (e.g., "What are the parts involved in memory?")

# Perform similarity search in the vectorstore
docs = semanticstore.similarity_search(user_input, k=VECTOR_RESULT_CNT)  # Fill in the vectorstore name and number of results

# Display retrieved documents
for i in docs:
    print("Source:", i.metadata['source'])  # Fill in the correct key for source (e.g., 'source')
    print("Page:", i.metadata['page'], "\n")  # Fill in the correct key for page number (e.g., 'page')
    print("Page Content:", i.page_content)
    print("---\n")

# Define metadata fields for structured retrieval
metadata_field_info = [
    AttributeInfo(
        name="page",  # Fill in the metadata field name (e.g., "Category")
        description="page number of document",  # Describe what this field represents
        type="integer"  # Fill in the data type (e.g., "string", "integer")
    ),
    AttributeInfo(
        name="source",
        description="file path of document",
        type="string"
    ),
    AttributeInfo(
        name="page_content",
        description="raw text of (sectional) document",
        type="string"
    )
]

# Describe the content of the document
document_content_description = "Text Semantic Chunks for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization"

structured_retriever = SelfQueryRetriever.from_llm(
    llm,
    semanticstore,
    document_content_description,
    metadata_field_info
)

# Test the filtering logic with Self-Retriever Queries
# Use queries that contain metadata filters: source, page, document type for semantic and vector stores.
SELF_RETRIEVER_QUERIES = [
    "What is the recommended dosage for scurvy?",
    "What is definition of Vitamin C but only from content found in the first quarter of the document?",
    "Describe the clinical signs of deficiency but exclude any data from the source `Pediatric Nutrition Guide.pdf`.",
    "What type of nutritional support is needed for patients to increase lean body mass?",
    "On page 35, please explain the correlation between Vitamin B12 levels and tissue deficiency.",
    "What are the laboratory and clinical standards for diagnosing Vitamin D deficiency? Specifically address adult patients."
]

# Query all SELF_RETRIEVER_QUERIES using the self-querying retriever
start_time = start_timer()

for ques in SELF_RETRIEVER_QUERIES:
    semantic_chunks_retrieved = structured_retriever.invoke(ques)
    print(f"Question: {ques}")
    print(f"Number of Semantic Chunks Retrieved: {len(semantic_chunks_retrieved)}")
    print(f"Retrieved Documents: {semantic_chunks_retrieved}")
    print("---\n")

show_timer(start_time)

# Define a prompt for generating hypothetical questions
hypothetical_questions_prompt = """
You are an AI {ROLE} specialized in generating precise, clinically relevant questions for information retrieval.
Your task is to analyze the provided TEXT CHUNK and generate a list of exactly three hypothetical questions for which the chunk contains the complete answer.

**QUESTION STYLE REQUIREMENTS:**
1. Questions must be factual and directly address **diagnostic criteria, treatment dosages, clinical findings, or defining concepts** mentioned in the TEXT CHUNK.
2. Phrasing must be natural and sound like a question a {ROLE} would actually ask.

TEXT CHUNK:
{docs}

{PROMPT_INSTR}
"""

# Generate hypothetical questions for tables in documents and store them as structured metadata.
start_time = start_timer()

collection_name = "hypothetical_questions"
rate_limit_hit = False
sleep_time = random.randint(25, 45)
batch_size = CHUNK_TEXT_BATCH_SIZE  # Process 50 chunks at a time

hypothetical_questions = []

for batch_start in range(0, len(semantic_chunks), batch_size):
    batch = semantic_chunks[batch_start: batch_start + batch_size]

    # List to store documents with hypothetical questions
    batched_hypothetical_questions = []

    for i, document in enumerate(batch, start=batch_start):
        try:
            # Invoke the LLM to generate questions based on the chunk content
            formatted_response = hypothetical_questions_prompt.format(
                ROLE=ROLE,
                PROMPT_INSTR=PROMPT_INSTR,
                docs=document.page_content
            )

            questions = filter_response(llm.invoke(formatted_response), i)

        except Exception as e:
            handle_rate_limit_error(e, collection_name, sleep_time)
            questions = EMPTY_RESP # Formerly "NA"

            sleep_time, rate_limit_hit = handle_rate_limit_error(e, collection_name, sleep_time)

        if rate_limit_hit:
            break

        # Only append metadata for successful responses
        if questions and questions != EMPTY_RESP:

            # Create metadata for the generated question
            questions_metadata = {
                'original_content': document.page_content, # Store the original chunk content
                'source': document.metadata['source'],     # Source document of the chunk
                'page': document.metadata['page'],         # Page number where the chunk appears
                'doc_type': collection_name,               # Indicate the content type
            }

            # Create and store the document containing generated questions
            batched_hypothetical_questions.append(
                create_document(
                    questions,
                    questions_metadata
                )
            )

    # Store each chunk into the master list of documents with hypothetical questions
    hypothetical_questions.extend(batched_hypothetical_questions)

    # ** Wait for 1 minute before processing the next batch **
    print(f"Processed {batch_start + batch_size} / {len(semantic_chunks)} chunks. Waiting {sleep_time} seconds...")
    time.sleep(sleep_time)

show_timer(start_time)

print(hypothetical_questions)

# Show a random sample of generated hypothetical questions
show_sample(hypothetical_questions, "Hypothetical Questions")

# Define a prompt for generating hypothetical questions for tables
hypothetical_questions_prompt = """
[SYSTEM INSTRUCTION]
You are an AI {ROLE} specialized in generating precise, clinically relevant questions for information retrieval.
Your task is to analyze the provided CONTEXT and TABLE DATA, and generate a list of three hypothetical questions that are directly answerable by the data.

[RULES]
1. Focus questions strictly on the **numeric data, specific values, formulas, or definitive lists** found in the table.
2. Use the **adjacent text (if provided)** to establish the clinical context for the questions.
3. If no adjacent text is provided, generate questions based on the table data alone.
4. **DO NOT** include any preamble, conversational text, or explanation in your response.
5. **OUTPUT ONLY THE JSON OBJECT.**

[FULL CONTEXT]
---TEXT ADJACENT TO TABLE---
{docs}

--- TABLE DATA (Structured for Analysis) ---
{tables}

[OUTPUT FORMAT]
Generate a JSON object containing a list of questions under the key "questions".
The list must contain **a minimum of 1 and a maximum of 3** questions.

{PROMPT_INSTR}
"""

# Generate hypothetical questions for tables in documents and store them as structured metadata.
start_time = start_timer()

# List to store documents with hypothetical questions for tables
rate_limit_hit = False
collection_name = "table_hypothetical_questions"
table_hypothetical_questions = []  # Initialize an empty list to store generated questions

# Generate hypothetical questions for each table in the documents
for document in tables:  # Iterate over all processed documents
    for page_number in tables[document]:  # Iterate over pages in the document
        table_in_page = tables[document][page_number]  # Extract the table from the document

        try:
            # Generate questions using the LLM based on the table content
            page_content_text = page_texts.get(document, {}).get(page_number, "")

            formatted_response = hypothetical_questions_prompt.format(
                ROLE=ROLE,
                docs=page_content_text,
                tables=table_in_page,
                PROMPT_INSTR=PROMPT_INSTR,
            )

            response = llm.invoke(formatted_response)
            questions = filter_response(response, page_number)

        except Exception as e:
            handle_rate_limit_error(e, collection_name, sleep_time)
            questions = EMPTY_RESP # Formerly "NA"

            sleep_time, rate_limit_hit = handle_rate_limit_error(e, collection_name, sleep_time)

        if rate_limit_hit:
            break

        if questions and questions != EMPTY_RESP:

            # Metadata for each table
            questions_metadata = {
                'original_content': str(table_in_page),  # Store the content of the original table
                'source': document,  # Store the source document name or identifier
                'page': page_number,  # Store the page number where the table was found
                'doc_type': collection_name,  # Indicate that the content type is a table
            }

            # Create a Document object for each set of generated questions
            table_hypothetical_questions.append(
                create_document(questions, questions_metadata)
            )

show_timer(start_time)

# Show a sample of generated hypothetical questions for tables
show_sample(table_hypothetical_questions, "Hypothetical Questions for Tables")

# Assign unique IDs to hypothetical questions and store them in the Chroma vector database in batches.
collection_name = "hypothetical_questions"

# Add IDs to the hypothetical questions for text semantic chunks
documents = hypothetical_questions

# Store the document chunks in Chroma vectorstore in batches
vectorstore = Chroma(
    embedding_function=embedding_model,
    collection_name=collection_name,
    persist_directory=format_dir("questions")
)

batch_size = CHUNK_DOC_BATCH_SIZE
for i in range(0, len(documents), batch_size):
    batch = documents[i : i + batch_size]
    vectorstore.add_documents(batch)

# Assign unique IDs to hypothetical questions for tables and store them in the Chroma vector database in batches.
collection_name = "table_hypothetical_questions"

# Add IDs to the hypothetical questions for tables
table_documents = table_hypothetical_questions

vectorstore = Chroma(
    embedding_function=embedding_model,
    collection_name="table_hypothetical_questions",
    persist_directory=format_dir("table_questions")
)

batch_size = CHUNK_DOC_BATCH_SIZE  # Define batch size for storing chunks (adjust as needed)
for i in range(0, len(table_documents), batch_size):
    batch = table_documents[i: i + batch_size]
    print(f"Added batch {i // batch_size + 1}. Total table documents processed: {i + len(batch)}")
    vectorstore.add_documents(batch)

# Mount Google Drive
drive.mount(GOOGLE_DRIVE_PATH)

# Define source and destination paths for vector storage
source_path = VECTORS_DIR  # Complete the code to define the path to your vectorstore directory
destination_path = GOOGLE_DRIVE_PATH + HOME_PATH + VECTORS_DIR  # Complete the code to define the destination path in your Drive

# Copy the directory to Google Drive
try:
    shutil.copytree(source_path, destination_path)
    print(f"Successfully copied '{source_path}' to '{destination_path}'")

except FileExistsError:
    print(f"Directory '{destination_path}' already exists. Skipping copy.")

except Exception as e:
    print(f"Error copying directory: {e}")

# Verify if the directory was copied successfully
if os.path.exists(destination_path):
    print(f"{source_path} directory exists in your Google Drive.")  # Complete the code to confirm the directory name

else:
    print(f"{source_path} directory was not copied to your Google Drive.")  # Complete the code to confirm the directory name
    os.makedirs(destination_path, exist_ok=True)
    print(f"Making the directory {destination_path} now...")

# Define metadata fields for structured retrieval of hypothetical questions
metadata_field_info = [
    AttributeInfo(
        name="original_content",
        description="Original text extracted from documents",
        type="string"
    ),
    AttributeInfo(
        name="source",
        description="File path of document",
        type="string"
    ),
    AttributeInfo(
        name="page",
        description="Page number of document",
        type="integer"
    ),
    AttributeInfo(
        name="type",
        description="Datatype of attribute `original_content`",
        type="string"
    )
]

# Fill in the document description, Describe the content of the document
document_content_description = "Hypothetical Questions for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization"

# Initialize the self-querying retriever for hypothetical questions
structured_hyp_retriever = SelfQueryRetriever.from_llm(
    llm,                           # LLM model
    vectorstore,                   # Vectorstore
    document_content_description,  # Document content description
    metadata_field_info            # Metadata field info
)

# Example user query
user_input = SELF_RETRIEVER_QUERIES[0]

# Perform structured retrieval using the self-querying retriever
hypothetical_questions_retrieved = structured_hyp_retriever.invoke(user_input)

# Print sample retrieved documents
print(hypothetical_questions_retrieved)

# Example user query
user_input = SELF_RETRIEVER_QUERIES[1]

# Perform structured retrieval using the self-querying retriever
hypothetical_questions_retrieved = structured_hyp_retriever.invoke(user_input)

# Print sample retrieved documents
print(hypothetical_questions_retrieved)

# As a specialist ask a all the questions and check the responses that have been vectored and stored.
start_time = start_timer()
for ques in SELF_RETRIEVER_QUERIES:
    hypothetical_questions_retrieved = structured_hyp_retriever.invoke(ques)
    print(f"\nQuestion: {ques}")
    print(f"Retrieved: {hypothetical_questions_retrieved}")
    print(f"Retrieved Count: {len(hypothetical_questions_retrieved)}")
    print("---\n")

show_timer(start_time)

import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module='jupyter_client',
    lineno=203
)

# Import necessary libraries
from typing import Dict, List, Tuple, Any  # Python typing for function annotations
from langchain_core.runnables import RunnablePassthrough  # LangChain core library for running pipelines
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text documents
from langchain_community.document_loaders import PyPDFLoader  # PDF document loader
from langchain_community.vectorstores import FAISS  # FAISS vector store
from langchain.embeddings.openai import OpenAIEmbeddings  # OpenAI embeddings for text vectors
from langchain.prompts import ChatPromptTemplate  # Template for chat prompts
from langchain_core.output_parsers import StrOutputParser  # String output parser
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END  # State graph for managing states in LangChain
from pydantic import BaseModel  # Pydantic for data validation
import numpy as np  # Numpy for numerical operations
from typing import TypedDict  # Typing for structured data

from langgraph.graph import END, StateGraph, START

from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from mem0 import MemoryClient

from IPython.display import Image, display

# @deprecated
# Define a function to read a JSON config file and return its contents as a dictionary.
def read_config(config_file):
  """Reads a JSON config file and returns a dictionary."""
  with open(config_file, 'r') as f:
    return json.load(f)

# @deprecated
# Read the configuration files
config = read_config("")  #Copy and paste the path of the config file uploaded in Colab
config2 = read_config("")

# Extract keys and endpoints from the configuration files

api_key = config.get("API_KEY")
endpoint = config.get("OPENAI_API_BASE")
llamaparse_api_key = config2.get("LLAMA_KEY")

# @deprecated
# Initialize the OpenAI embedding function for Chroma
embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_base=_________,  # Fill in the API base URL
    api_key=_________,  # Fill in the API key
    model_name='____________'  # Fill in the model name
)
# This initializes the OpenAI embedding function for the Chroma vectorstore, using the provided endpoint and API key.


# @deprecated
# Initialize the OpenAI Embeddings
embedding_model = OpenAIEmbeddings(
    openai_api_base=_________,  # Fill in the endpoint
    openai_api_key=_________,  # Fill in the API key
    model=''                 # Fill in the model name
)
# This initializes the OpenAI embeddings model using the specified endpoint, API key, and model name.

# @deprecated
# Initialize the Chat OpenAI model
llm = ChatOpenAI(
    base_url=_________,  # Fill in the endpoint
    openai_api_key=_________,  # Fill in the API key
    model="gpt-4o-mini",  # Fill in the deployment name (e.g., gpt-4o-mini)
    streaming=False
)
# This initializes the Chat OpenAI model using the provided endpoint, API key, deployment name.

# @deprecated
# set the LLM and embedding model in the LlamaIndex settings.
Settings.llm = llm
Settings.embedding = embedding_model

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

show_logs = True

def expand_query(state: AgentState) -> AgentState:
    """
    Expands the user query to improve retrieval of nutrition-disorder-related information using few-shot prompting.

    Args:
        state (Dict): The current state of the workflow, containing the user query.

    Returns:
        Dict: The updated state with the expanded query.
    """

    if show_logs:
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
        if show_logs:
            print("---Using feedback to refine query---")

        system_message += f"""

        You have already generated a query that was not precise enough. Use the following SUGGESTIONS to create a NEW, improved query.

        SUGGESTIONS:
        {query_feedback}
        """

    # Create the final prompt template
    expand_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message.strip()),
        ("user", "Original User Query: {query}")
    ])

    chain = expand_prompt | llm | StrOutputParser()

    # Invoke the chain
    state['expanded_query'] = chain.invoke({
        "query": original_query,
        # Note: Feedback is injected via the system_message,
        # but we need to pass a clean 'query' and 'ROLE'
        "ROLE": state['ROLE'],
    })

    # Clear the feedback for the next node
    state['query_feedback'] = ""

    return state

# Initialize the Chroma vector store for retrieving documents
collection_name = "nutritional"

vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=format_dir(collection_name),
    embedding_function=embedding_model
)

# Create a retriever from the vector store
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': VECTOR_RESULT_CNT}
)

# It creates documents where the page_content is the full, factual text from your PDF.
# We will now add table question documents to the latest vector storage.  This version has page_texts and tables as a source from the original document.
start_time = start_timer()

orig_docs = []
for file_name, text_dict in page_texts.items():
    for page_number, text_content in text_dict.items():

        # Retrieve the table data for this specific page (if it exists)
        table_data = tables.get(file_name, {}).get(page_number, [])

        # Format table rows into a readable string
        table_string = "\n".join([" | ".join(map(str, row)) for row in table_data])

        # Combine the text content and the table data
        full_content = (
            f"Context from Page {page_number}:\n{text_content}\n"
            f"Associated Table Data:\n{table_string}"
        )

        # Create a Document object
        orig_docs.append(create_document(
            content=full_content.strip(),
            metadata={
                "source": file_name,
                "page": page_number
            }
        ))

show_timer(start_time)

print(orig_docs)

start_time = start_timer()

print(f"Indexing {len(orig_docs)} original documents into '{collection_name}'...")
vector_store.add_documents(orig_docs)
print("Indexing of original documents complete! You can now query the RAG agent.")

show_timer(start_time)

def retrieve_context(state: AgentState) -> AgentState:
    """
    Retrieves context from the vector store using the expanded or original query.

    Args:
        state (Dict): The current state of the workflow, containing the query and expanded query.

    Returns:
        Dict: The updated state with the retrieved context.
    """

    query = state['expanded_query']

    if show_logs:
        print("\n-------- retrieve_context --------")
        print("Query used for retrieval:", query)  # Debugging: Print the query

    # Retrieve documents from the vector store
    retrieved_docs = retriever.invoke(query)

    if show_logs:
        print("Retrieved documents:", retrieved_docs)  # Debugging: Print the raw docs object

    # Extract both page_content and metadata from each document
    state['context'] = [
        {
            "content": doc.page_content,  # The actual content of the document
            "metadata": doc.metadata  # The metadata (e.g., source, page number, etc.)
        }
        for doc in retrieved_docs
    ]

    if show_logs:
        print("Extracted context with metadata:", state['context'])  # Debugging: Print the extracted context

    return state

def craft_response(state: Dict) -> Dict:
    """
    Generates a response using the retrieved context, focusing on nutrition disorders.

    Args:
        state (Dict): The current state of the workflow, containing the query and retrieved context.

    Returns:
        Dict: The updated state with the generated response.
    """

    if show_logs:
        print("\n-------- craft_response --------")

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

    response_prompt = ChatPromptTemplate.from_messages([
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

    if show_logs:
        print("intermediate response: ", response)

    return state

def score_groundedness(state: Dict) -> Dict:
    """
    Checks whether the response is grounded in the retrieved context.

    Args:
        state (Dict): The current state of the workflow, containing the response and context.

    Returns:
        Dict: The updated state with the groundedness score.
    """

    if show_logs:
        print("\n-------- check_groundedness --------")

    system_message = """You are a meticulous AI {ROLE} Quality Analyst and fact-checker. Your sole task is to evaluate how well a given response is supported by a provided context.
    Calculate a score from 0.0 to 1.0 that represents the fraction of claims in the response that are directly and verifiably supported by the context.
    - A score of 1.0 means every claim in the response is fully supported by the context.
    - A score of 0.0 means no claims in the response are supported by the context.

    **Your output MUST be the numerical score as a single float. Do not output any other text, explanation, or markdown.**
    """

    groundedness_prompt = ChatPromptTemplate.from_messages([
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

    if show_logs:
        print("groundedness_score: ", groundedness_score)
        print("######## Groundedness Incremented ##########")

    state['groundedness_score'] = groundedness_score

    return state

def check_precision(state: Dict) -> Dict:
    """
    Checks whether the response precisely addresses the user’s query.

    Args:
        state (Dict): The current state of the workflow, containing the query and response.

    Returns:
        Dict: The updated state with the precision score.
    """

    if show_logs:
        print("\n-------- check_precision --------")

    system_message = """
    As an AI {ROLE} evaluate whether the response precisely addresses the user's query.
    Evaluate, assign and return the precision score for the response.  Your evaluation is based solely on the relationship between the response and the query. Do not consider anything else.

    The score is from 0.0 (least) to 1.0 (best).
    - A score of 1.0 means the response is precise, on-topic and accurately addressed by the query.
    - A score of 0.0 means the response is ambiguous, off-topic, or inaccurate and does not accurately address the query.

    **Only output the numerical score as a float and nothing else!**
    """

    precision_prompt = ChatPromptTemplate.from_messages([
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

    if show_logs:
        print("precision_score:", precision_score)
        print("######## Precision Incremented ##########")

    return state

def refine_response(state: Dict) -> Dict:
    """
    Suggests improvements for the generated response.

    Args:
        state (Dict): The current state of the workflow, containing the query and response.

    Returns:
        Dict: The updated state with response refinement suggestions.
    """

    if show_logs:
        print("\n-------- refine_response --------")

    system_message = """
    You are an AI {ROLE} Quality Analyst and Critic. Your sole task is to provide constructive feedback on a given response based on the user's original query.
    Your feedback should identify potential gaps, ambiguities, or missing details and suggest specific improvements to enhance the response's accuracy and completeness.

    - Use bullet points to structure your suggestions.
    - Do NOT rewrite the full response. Only provide a list of suggestions for improvement.
    - Your output must be only the bulleted list of suggestions. Do not include a preamble like "Here are my suggestions:".
    """

    refine_response_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nResponse: {response}\n\n"
                 "What improvements can be made to enhance accuracy and completeness?")
    ])

    chain = refine_response_prompt | llm | StrOutputParser()

    # Store response suggestions in a structured format
    feedback = f"Previous Response: {state['response']}\nSuggestions: {chain.invoke({'query': state['query'], 'response': state['response'], 'ROLE': state['ROLE']})}"

    if show_logs:
        print("feedback: ", feedback)
        print(f"State: {state}")

    state['feedback'] = feedback

    return state

def refine_query(state: Dict) -> Dict:
    """
    Suggests improvements for the expanded query, returning them in a structured JSON format.

    Args:
        state (Dict): The current state of the workflow, containing the query and expanded query.

    Returns:
        Dict: The updated state with JSON-formatted query refinement suggestions.
    """

    if show_logs:
        print("\n--- refine_query ---")

    # Define the desired JSON schema for the output
    suggestion_schema = {
        "type": "object",
        "properties": {
            "missing_keywords": {
                "type": "array",
                "description": "List of technical or clinical keywords missing from the expanded query."
            },
            "scope_refinements": {
                "type": "array",
                "description": "List of suggestions to narrow or broaden the search scope (e.g., 'limit to pediatric patients' or 'include systematic reviews')."
            },
            "term_clarifications": {
                "type": "array",
                "description": "List of ambiguous terms in the expanded query that should be clarified or replaced with synonyms."
            }
        }
    }

    # This prompt forces the JSON structure and ensures high-quality clinical input
    system_message = f"""
    You are an AI Search Query Analyst specializing in clinical nutrition literature.
    Your sole task is to provide constructive feedback on the provided expanded query to enhance its search precision for academic databases.

    - Do NOT rewrite the query.
    - Analyze the Expanded Query against the Original Query and suggest improvements only in the required JSON format.
    - If a category has no suggestions, return an empty list for that key.

    - Your output MUST be a JSON object that strictly adheres to the following schema: {suggestion_schema}
    """

    # Use the LangChain JsonOutputParser for reliable structured output
    json_parser = JsonOutputParser(pydantic_object=suggestion_schema)

    refine_query_prompt = ChatPromptTemplate.from_messages([
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

    if show_logs:
        print(f"Query Feedback Generated (JSON):\n{suggestions_str}")

    return state

# Checks if the maximum number of iterations has been reached
def is_max_iterations_reached(state: Dict, var: str) -> bool:
    return state[var] >= state["loop_max_iter"]

def should_continue_groundedness(state):

  """Decides if groundedness is enough or needs improvement."""

  if show_logs:
      print("\n-------- should_continue_groundedness --------")
      print("groundedness loop count: ", state['groundedness_loop_count'])

  if state["groundedness_score"] >= EVAL_THRESHOLD:  # Threshold for groundedness
      if show_logs:
          print("Moving to precision")

      return "check_precision"
  else:
      if is_max_iterations_reached(state, "groundedness_loop_count"):
            return "max_iterations_reached"
      else:
          if show_logs:
              print(f"-------- Groundedness Score Threshold Not met. Refining Response ----------")

          return "refine_response"

def should_continue_precision(state: Dict) -> str:

    """Decides if precision is sufficient or needs improvement."""

    if show_logs:
        print("\n-------- should_continue_precision --------")
        print("precision loop count: ", state['precision_loop_count'])

    if state["precision_score"] >= EVAL_THRESHOLD:  # Threshold for precision
        return "pass"  # Complete the workflow
    else:
        if is_max_iterations_reached(state, "precision_loop_count"):  # Maximum allowed loops
            return "max_iterations_reached"
        else:
            if show_logs:
                print(f"-------- Precision Score Threshold Not met. Refining Query ----------")  # Debugging

            return "refine_query"  # Refine the query

def max_iterations_reached(state: AgentState) -> AgentState:
    """Handles the case where max iterations are reached."""
    state['response'] = "We need more context to provide an accurate answer."
    return state

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

WORKFLOW_APP = create_workflow().compile()

display(Image(WORKFLOW_APP.get_graph().draw_mermaid_png()))

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

# Queries used by the specialist.
# Tests the reasoning & synthesis loop
AGENTIC_RAG_QUERIES = [
    "What are the diagnostic criteria and the best source of vitamin C for scurvy?",
    "List the components of Macronutrients as described in the reference.",
    "What is the prevalence of Protein-Energy Undernutrition (PEU) in developed countries, and how does it correlate with patient age?",
    "What are the primary preventative measures against the development of Vitamin D deficiency and dependency?",
    "Describe the clinical manifestations and the specific recommended dosage for treatment of severe acute Zinc deficiency.",
    "According to the reference, what is the definition of Obesity, and what are the associated complications and measured levels used for diagnosis?"
]

# Questions for the RAG Agent
start_time = start_timer()
for ques in AGENTIC_RAG_QUERIES:
    ans = agentic_rag.invoke(ques)
    print(f"Question: {ques}")
    print(f"Answers: {ans}")
    print(f"Answer Count: {len(ans)}")
    print("---\n")

show_timer(start_time)

# Import libraries for handling user input filtering and accessing user data
from groq import Groq  # Llama Guard client for filtering user input

# Initialize the Llama Guard client with the API key
llama_guard_client = Groq(api_key=GROQ_API_KEY)

# Define prompt messages and queries
"""
--- Llama Guard 3 8B for S14 Code Interpreter Abuse ---
see: https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3

S1:  Violent Crimes.
S2:  Non-Violent Crimes.
S3:  Sex Crimes.
S4:  Child Exploitation.
S5:  Defamation.
S6:  Specialized Advice.
S7:  Privacy.
S8:  Intellectual Property.
S9:  Indiscriminate Weapons.
S10: Hate.
S11: Self-Harm.
S12: Sexual Content.
S13: Elections.
S14: Code Interpreter Abuse

* We will permit codes S6, S7, S8, and S13 for this Nutrition Disorder Specialist bot. *
"""
LLAMA_UNSAFE_CODES = ["S1", "S2", "S3", "S4", "S5", "S9", "S10", "S11", "S12"]

# Function to filter user input with Llama Guard
def filter_input_with_llama_guard(user_input_str: str, model=LLAMA_MODEL) -> str:
    """
    Filters user input using Llama Guard to ensure it is safe.
    Whitelist "UNSAFE" codes: S6, S7, S8, S13 so that you can handle the customer query.
    Example of an unsafe result: UNSAFE S10, S11

    Parameters:
    - user_input: The input provided by the user.
    - model: The Llama Guard model to be used for filtering (default is "meta-llama/llama-guard-4-12b").

    Returns:
    - The filtered and safe input.
    """

    try:
        # Create a request to Llama Guard to filter the user input
        llama_response = llama_guard_client.chat.completions.create(
            messages=[{"role": "user", "content": user_input_str}],
            model=model,
        )

        # Return the filtered input
        result = llama_response.choices[0].message.content.strip()

        if "UNSAFE" in result:
            if any(code.strip() in LLAMA_UNSAFE_CODES for code in result.replace("UNSAFE ", "").strip().split(",")):
                return "BYPASS_SAFE"
            else:
                return "UNSAFE"
        else:
            return "SAFE"

    except Exception as e:
        print(f"Error with Llama Guard: {e}")
        return ""

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
            base_url=OPENAI_API_BASE,
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
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),  # System instructions
            ("human", "{input}"),  # Placeholder for human input
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
            context += f"Customer: {memory['memory']}\n" # Customer's past messages
            context += f"Support: {memory['memory']}\n"  # Chatbot's past responses
            context += "---\n"

        # Print context for debugging purposes
        if show_logs:
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

def nutrition_disorder_agent():
    """
    A conversational agent that answers nutrition-disorder-related questions
    using a RAG-based workflow with safety filtering and user session handling.
    """

    print("|--------------------------------------------------------------------")
    print(f"| {TITLE}")
    print("|--------------------------------------------------------------------")
    print("| Welcome! I'm your dedicated AI Nutrition Agent.")
    print("| Ask me anything about nutrition disorders. You can inquire about\n| symptoms, causes, treatment options, or preventative measures.\n| I'm ready to help with your health-related questions.")
    print("|")
    print(f"| Type '{EXIT_CMD}' to end the conversation.")
    print("|--------------------------------------------------------------------\n")

    chatbot = NutritionBot()  # Initialize chatbot instance
    chatbot.agent_executor.verbose = set_agent_logs()  # Set logging preferences

    # This provides a way to initiate a chat as different users.
    user_id = input("Agent: Login by providing customer name ")  # Get user ID for tracking conversation sessions

    print(f"\n--- Session Start: {show_datetime()} ---\n")

    while True:
        # Get user input
        print("Agent: How can I help you?\n")
        user_query = input(f"{user_id}: ")

        # Set timer for each question
        q_time = start_timer()

        # Define the logic for exiting the loop' [if the user types in exit]
        if user_query.lower() == EXIT_CMD:
            print("\nAgent: Goodbye! Feel free to return if you have more questions.")
            print(f"--- Session End: {show_datetime()} ---")
            break

        # Filter input through Llama Guard - returns "SAFE" or "UNSAFE"
        filtered_result = filter_input_with_llama_guard(user_query)  # Call function to filter input
        filtered_result = filtered_result.replace("\n", " ")  # Normalize the result

        # Check if filtered_result is SAFE or UNSAFE
        if filtered_result in ["SAFE", "BYPASS_SAFE"]:
            # Process the user query using the RAG workflow
            try:
                response = chatbot.handle_customer_query(user_id, user_query)  # Call chatbot handler function
                print(f"Agent: {response}\n")

            except Exception as e:
                print("Agent: Sorry, I encountered an error while processing your query. Please try again.")
                print(f"Customer Query Error: {e}\n")
        else:
            print(f"Agent: I apologize, but I cannot process that input `{filtered_result}` as it may be inappropriate. Please try again.")

        # Show answer duration per query
        print(f"[Answered in {get_time(q_time)}]\n")


# RUN THE AI AGENT
start_time = start_timer()

nutrition_disorder_agent()

show_timer(start_time)

%%writefile app.py

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

from dotenv import load_dotenv  # Loading environment variables from a .env file
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

# --- DEFINE CONFIGURATIONS AND CONSTANTS
# Note: os.getenv() are the secrets defined in the Huggingface.co settings page.
# os.getenv() is for READING a variable from the operating system's environment.

# Hugging Face
# see: https://hugginface.co
# see: Model -> https://huggingface.co/jasonmonroe/smart-nutri-disorder-specialist-model
# see: Space -> https://huggingface.co/jasonmonroe/smart-nutri-disorder-specialist-bot
HF_REPO_ID = "jasonmonroe/smart-nutri-disorder-specialist-bot"
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
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # embedding models "text-embedding-ada-002", "text-embedding-3-large"
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
    hours, remainder = divmod(diff, (SECS_IN_MIN*SECS_IN_MIN))
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
    st.session_state[session_keys_valid] = None

if session_doc_found not in st.session_state:
    st.session_state[session_doc_found] = None

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
print("--- START PROGRAM ---")

# --- FILTER INPUT WITH LLAMA GUARD
# Initialize the Llama Guard client with the API key
llama_guard_client = Groq(api_key=GROQ_API_KEY)

# Initialize the OpenAI Embeddings
# see: https://docs.langchain.com/oss/python/integrations/text_embedding/openai
embedding_model = OpenAIEmbeddings(
    openai_api_base=OPENAI_API_BASE, # Fill in the endpoint
    openai_api_key=OPENAI_API_KEY,   # Fill in the API key
    model=OPENAI_EMBEDDING_MODEL,          # Fill in the model name
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


%%writefile requirements.txt

######################## WRITE YOUR SCRIPT HERE  #########################
numpy==1.26.4
openai==1.55.3
langchain==0.2.7
langchain-community==0.2.7
langchain-huggingface==0.0.3
langchain-experimental==0.0.62
langchain-openai==0.1.14
chromadb==0.5.3
sentence-transformers==3.0.1
python-dotenv==1.0.1
lark==1.1.9
llama-index-core
llama-parse==0.5.11
llama-index-readers-file
llama-index-llms-langchain
langgraph
groq
mem0ai
streamlit==1.35.0

# Optional: Backup requirements to Google Drive
!pip install pipreqs

project_path = GOOGLE_DRIVE_PATH + HOME_PATH
!pipreqs "{project_path}" --force

%%writefile Dockerfile

######################## WRITE YOUR DOCKERFILE SCRIPT HERE  #########################
#
# Dockerfile based on best practices for Streamlit on Hugging Face Spaces
#
# Use a stable, slim Python base image
FROM python:3.11-slim

# 1. Install system dependencies (git and git-lfs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 1000 user

# Set the active user for all subsequent steps
USER user

# Ensure user's local installation directory is on the PATH
ENV PATH="/home/user/.local/bin:$PATH"

# Set application working directory
WORKDIR /app

# Copy requirements and install Python dependencies
# Specify user:group ownership for robustness
COPY --chown=user:user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application files
COPY --chown=user:user . /app

# 4. Download Large File Storage (LFS) assets
# This is crucial for models or large data files
RUN git lfs pull

# Command to run the Streamlit application
# Use --server.address=0.0.0.0, as port 7860 is the default for Spaces
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

from huggingface_hub import HfApi

# Add your HF_TOKEN credential into your Colab secrets
api = HfApi(token=HF_TOKEN)

api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=HF_REPO_ID,
    repo_type="space",
    commit_description="Ran requirements cell to create file and uploading it to HuggingFace.co"
)

api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id=HF_REPO_ID,
    repo_type="space",
    commit_description="Ran app.py cell to create file and uploading it to HuggingFace.co"
)

api.upload_folder(
    folder_path="vectorstore/nutritional_db",
    path_in_repo="nutritional_db",
    repo_id=HF_REPO_ID,
    repo_type="space",

    # Optional: Add a short commit message
    commit_description="Pushing nutritional database from vector storage and uploading it to HuggingFace.co"
)

# Upload Dockerfile
api.upload_file(
    path_or_fileobj="Dockerfile",
    path_in_repo="Dockerfile",
    repo_id=HF_REPO_ID, # Replace it with your username and space name
    repo_type="space",
    commit_description="Ran Dockerfile cell to create file and uploading it to HuggingFace.co"
)

# Upload Documents
api.upload_file(
    path_or_fileobj=DOCUMENT_ZIP,
    path_in_repo=DOCUMENT_ZIP,
    repo_id=HF_REPO_ID,
    repo_type="space",
    commit_message="Add data zip file for Dockerfile to unzip.",
    commit_description="Uploading the compressed medical reference data to be unzipped during the Docker image build process.",
)
