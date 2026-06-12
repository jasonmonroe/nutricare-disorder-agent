# src/config.py

# +----------------+
# |     CONFIG     |
# +----------------+

from dotenv import load_dotenv

load_dotenv()

import os
import random

# --- DEFINE CONFIGURATIONS AND CONSTANTS --- #
# Note: os.getenv() are the secrets defined in the Huggingface.co settings page.
# os.getenv() is for READING a variable from the operating system's environment.

# Groq
# see: https://www.groq.com/
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Hugging Face
# see: https://hugginface.co
# see: Model -> https://huggingface.co/jasonmonroe/smart-nutri-disorder-specialist-model
# see: Space -> https://huggingface.co/jasonmonroe/smart-nutri-disorder-specialist-bot
# Note: Make sure you have your own Huggingface Repo ID!
HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

# Llama
# see: https://llama.developer.meta.com/docs/api-keys/
LLAMA_KEY = os.getenv("LLAMA_KEY")  # Fill in your Llama API key, Used for LlamaParse()
# see: https://huggingface.co/openai/gpt-oss-safeguard-20b
# see: https://openai.com/index/introducing-gpt-oss-safeguard
# Note: If Llama Model is defunct, see: https://console.groq.com/docs/deprecations
LLAMA_MODEL = os.getenv("LLAMA_MODEL")

# Mem0
# see: https://mem0.ai
MEM0_API_KEY = os.getenv("MEM0_API_KEY")  # Fill in your Mem0 API key

# OpenAI
# see: https://openai.com/api/
# see: https://developers.openai.com/api/docs/models/text-embedding-3-small
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # Fill in the OpenAI API base URL (e.g., "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Fill in your OpenAI API Token (from your account)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")  # embedding models "text-embedding-ada-002", "text-embedding-3-large"
OPENAI_MODEL = os.getenv("OPENAI_MODEL")  # Fill in the OpenAI model name (e.g., "gpt-4o-mini")

# Agents
AGENT_EVAL_THRESHOLD = 0.8
AGENT_EMPTY_RESP = "[]" # Empty response
AGENT_EXIT_CMD = "exit"
AGENT_WORKFLOW_IMAGE = "outputs/graph_workflow.png"
AGENT_RETRIEVAL_LIMIT = 5

# Define document directory paths and chunk sizes
# Batch sizes (per batch) for processing documents and text chunks
DOCUMENT_CHUNK_BATCH_SIZE = 100
DOCUMENT_CHUNK_TEXT_BATCH_SIZE = 50
DOCUMENT_DIR = "data/nutritional-medical-reference"
DOCUMENT_FILE = 'nutritional-disorders.pdf'
DOCUMENT_FILEPATH = DOCUMENT_DIR + '/' + DOCUMENT_FILE
DOCUMENT_ZIP = "data/nutritional-medical-reference.zip" # Zip file name

CHROMA_SERVER_NO_TELEMETRY = "true"
CHROMA_TELEMETRY_DISABLED = "1"
RATE_LIMIT_TIME = 8 #random.randint(3, 7) # was, 1,3. Increased for free tier Groq API
SLEEP_TIME_INC = 0.20 # was 0.15. Increased for more aggressive backoff
SEMANTIC_THRESH_LIMIT = random.randint(80, 95)
VECTOR_RESULT_CNT = random.randint(4, 5) #3
VECTORS_DIR= "db/"

# Used for searching the document for pairing subject for building an effective RAG system.
# This query asks a specific question about vitamin deficiencies and memory impairment.
# Match keywords and map them with embedding models.
SIMILARITY_SEARCH_QUERY = "What nutritional deficiency, such as folate deficiency or that caused by alcoholism, is clinically linked to anemia, and what specific standard diagnostic metric is used for its confirmation?"

# --- Prompt variables ---
PROMPT_INSTR = """
    Important:
    Generate only a Python list of relevant questions (e.g., ['Question 1', 'Question 2']).
    *Do NOT mention or output anything before or after the list, including commentary, markdown blocks, or extra punctuation.*
    If the content cannot answer any question(s), your output MUST be the empty Python list: [].
    """.strip()

AI_ROLE = "Nutrition Disorder Specialist"
AI_TITLE = "SMART NUTRITION DISORDER SPECIALIST BOT"
APP_TITLE = 'NUTRICARE DISORDER AGENT'

"""
Define prompt messages and queries

Llama Guard 3 8B for S14 Code Interpreter Abuse
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

We will permit codes S6, S7, S8, and S13 for this Nutrition Disorder Specialist bot.
"""
LLAMA_UNSAFE_CODES = ["S1", "S2", "S3", "S4", "S5", "S9", "S10", "S11", "S12"]
LLAMA_SAFE = ["SAFE", "BYPASS_SAFE"]

# Miscellaneous constants
MSEC = 1000
SECS_IN_MIN = 60 # secs in min
INACTIVE_SESSION_DUR = SECS_IN_MIN * 5 # 5  minutes 
RUN_MIN_ID = 10000
RUN_MAX_ID = 99999


# Icons
I_ANGRY = '😠'
I_BOT = '🤖'
I_BOOK = '📚'
I_BROOM = '🧹'
I_CHECKMARK = '✅'
I_CLOCK = '⏰'
I_CONFUSED = '😕'
I_CROSSMARK = '❌'
I_DB = '📊'
I_DIR = '📂'
I_DEAD = '😵'
I_DISK = '💾'
I_DOCUMENT = '📄'
I_EXCLAMATION = '❗'
I_FIRE = '🔥'
I_FLAG = '🚩'
I_FROWN = '😦'
I_GEAR = '⚙️'
I_GHOST = '👻'
I_HANDSHAKE = '🤝🏾'
I_INFO = 'ℹ️'
I_MINUS = '➖'
I_PEN = '🖊️'
I_PLUS = '➕'
I_QUES = '❓'
I_RUNNING = '🏃'
I_SAD = '😢'
I_SKULL = '💀'
I_SLEEPING = '😴'
I_SMILING = '😊'
I_STAR = '⭐'
I_SURPRISED = '😲'
I_TIMER = '⏱'
I_THINKING = '🤔'
I_THUMBS_DOWN = '👎'
I_THUMBS_UP = '👍'
I_WARNING = '⚠️'
I_WATCH = '⌚'
