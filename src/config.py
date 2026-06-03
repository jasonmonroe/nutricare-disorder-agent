# src/config.py

import os

# --- CONSTANTS ---- #
APP_TITLE = 'Nutricare Disorder Agent'

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
OPENAI_EMB_MODEL = "text-embedding-3-small"  # embedding models "text-embedding-ada-002", "text-embedding-3-large"
OPENAI_MODEL = "gpt-4o-mini"  # Fill in the OpenAI model name (e.g., "gpt-4o-mini")

EVAL_THRESHOLD = 0.8
EXIT_CMD = "exit"

RETRIEVAL_LIMIT = 5

VECTOR_RESULT_CNT = 3
VECTOR_RESULT_CNT = 3

# Define the Google Drive and other directory paths
COLLECTION_NAME = "nutritional"
DOCUMENT_DIR = "Nutritional Medical Reference"
DOCUMENT_ZIP = "Nutritional_Medical_Reference.zip" # Zip file name

# Prompt variables
AI_ROLE = "Nutrition Disorder Specialist"
AI_TITLE = "SMART NUTRITION DISORDER SPECIALIST BOT"

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


# Miscellaneous constants
MILLI_IN_SECS = MSEC = 1000
SECS_IN_MIN = 60 # secs in min
MIN_RUN_ID=10000
MAX_RUN_ID=99999


# Icons
I_ANGRY = '😠'
I_BOT = '🤖'
I_BOOK = '📚'
I_CHECKMARK = '✅'
I_CLOCK = '⏰'
I_CONFUSED = '😕'
I_CROSSMARK = '❌'
I_DB = '📊'
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
I_PAPER = '📄'
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
I_THINKING = '🤔'
I_THUMBS_DOWN = '👎'
I_THUMBS_UP = '👍'
I_WARNING = '⚠️'
I_WATCH = '⌚'
