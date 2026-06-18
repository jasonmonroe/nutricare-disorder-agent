# src/constants_premium.py

# +----------------------+
# |      CONSTANTS       |
# |    Premium Models    |
# +----------------------+

"""
Premium Constants are for models used in Google Colab with premium models.
Use the model_config to load these.

LLAMA_MODEL=meta-llama/llama-guard-4-12b
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MODEL=gpt-4o-mini
"""

# Vendor Libraries
from langchain_classic.chains.query_constructor.schema import AttributeInfo

from src.constants import DOCUMENT_FILEPATH

VERSION = 'premium'

# --- Models --- #
PREMIUM_LLAMA_MODEL = 'meta-llama/llama-guard-4-12b'
PREMIUM_OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'
PREMIUM_OPENAI_MODEL = 'gpt-4o-mini'

CHROMA_VECTORS_DIR = "premium_chroma_db"

DOCUMENT_CHUNK_BATCH_SIZE = 100
DOCUMENT_CHUNK_TEXT_BATCH_SIZE = 50

RATE_LIMIT_TIME = 8
SEMANTIC_THRESH_LIMIT = 95  # Strict percentile boundary for high-precision chunks
SLEEP_TIME_SEC = 2
SLEEP_TIME_INC = 0.20 

# --- Prompt Formats ---
PROMPT_INSTR = """
[OUTPUT FORMAT REQUIREMENT]
Return a valid JSON array containing exactly three strings representing the generated questions. 
Do not include any conversational preamble, intro text, markdown formatting, or triple backticks. Start your response directly with the opening square bracket.

Example Response:
[
  "Factual diagnostic question 1?",
  "Factual treatment dosage question 2?",
  "Clinical finding question 3?"
]
""".strip()

PROMPT_QUESTION_GENERATOR = """
You are an AI {AI_ROLE} specialized in generating precise, clinically relevant questions for information retrieval.
Your task is to analyze the provided TEXT CHUNK and generate a list of exactly three hypothetical questions for which the chunk contains a complete, definitive answer.

[QUESTION STYLE REQUIREMENTS]
1. Questions must be highly factual and directly address diagnostic criteria, treatment dosages, clinical findings, or defining concepts mentioned in the text block.
2. Phrasing must sound natural and match the language a professional {AI_ROLE} would use in clinical practice.

[INPUT DATA]
TEXT CHUNK:
{docs}

{PROMPT_INSTR}
""".strip()

PROMPT_TABLE_QUESTION_GENERATOR = """
You are an AI {AI_ROLE} specialized in generating precise, clinically relevant questions for information retrieval from structured layouts.
Your task is to analyze the provided CONTEXT text and its accompanying TABLE DATA to extract exactly 3 high-quality clinical queries.

[RULES]
1. Focus questions strictly on the numeric data, specific diagnostic values, medical formulas, or definitive classifications found in the table matrix.
2. Use the adjacent text context to anchor and ground medical terminology.

[INPUT DATA]
--- TEXT ADJACENT TO TABLE ---
{docs}

--- TABLE DATA ---
{tables}

{PROMPT_INSTR}
""".strip()

# --- Self Retrievers --- #
RETRIEVER_DOCUMENT_CONTENT = f"Text Semantic Chunks published by the Global Nutritional Health Organization"
RETRIEVER_DOCUMENT_METADATA_FIELDS = [
    AttributeInfo(
        name="source",
        description="The exact file path name of the medical reference text. Use strictly for matching string values.",
        type="string"
    ),
    AttributeInfo(
        name="page",
        description="The physical page number within the PDF document. Must use strictly integer values for numerical comparison operations.",
        type="integer"
    ),
    AttributeInfo(
        name="doc_type",
        description="The explicit data classification marker specifying if a chunk is 'source_text', 'hypothetical_questions', or 'table_hypothetical_questions'.",
        type="string"
    )
]

# Used by Table Questions
RETRIEVER_HYPER_DOCUMENT_CONTENT = "Hypothetical clinical research queries derived from the reference workbook matching metadata vectors"
RETRIEVER_HYPER_DOCUMENT_METADATA_FIELDS = [
    AttributeInfo(
        name="original_content",
        description="Original text segment or structured table markdown layout string that anchors the generation.",
        type="string"
    ),
    AttributeInfo(
        name="source",
        description="The exact string file path of the source document.",
        type="string"
    ),
    AttributeInfo(
        name="page",
        description="The target physical page number index within the medical document. Use strictly integer values.",
        type="integer"
    ),
    AttributeInfo(
        name="doc_type",
        description="The explicit data classification marker specifying if a chunk is 'hypothetical_questions' or 'table_hypothetical_questions'.",
        type="string"
    )
]

RETRIEVER_QUERIES = [
    "What is the recommended dosage for scurvy?",
    "What is definition of Vitamin C but only from content found in the first quarter of the document?",
    f"Describe the clinical signs of nutritional deficiency documented in `{DOCUMENT_FILEPATH}`.",  # 👈 Fixed adversarial exclusion bug
    "What type of nutritional support is needed for patients to increase lean body mass?",
    "On page 35, please explain the correlation between Vitamin B12 levels and tissue deficiency.",
    "What are the laboratory and clinical standards for diagnosing Vitamin D deficiency? Specifically address adult patients."
]
