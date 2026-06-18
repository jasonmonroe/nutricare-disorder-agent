# src/constants_free.py

# +-------------------+
# |     CONSTANTS     |
# |    Free Models    |
# +-------------------+

"""
Free Constants are for models used in Python on a local MAC OS X.  These are freemium
cheap models that do NOT perform as well as premium.  It's to be used for evaluation 
purposes only.

Use the model_config to load these.

LLAMA_MODEL=llama-3.3-70b-versatile
OPENAI_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OPENAI_MODEL=llama-3.1-8b-instant
"""

# Vendor Libraries
from langchain_classic.chains.query_constructor.schema import AttributeInfo

# Local Libraries
from src.constants import DOCUMENT_FILE, DOCUMENT_FILEPATH

VERSION = 'free'

# --- Models --- #
FREE_LLAMA_MODEL = 'llama-3.3-70b-versatile'
FREE_OPENAI_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
FREE_OPENAI_MODEL = 'llama-3.1-8b-instant'

CHROMA_VECTORS_DIR = "free_chroma_db"

DOCUMENT_CHUNK_BATCH_SIZE = 32 
DOCUMENT_CHUNK_TEXT_BATCH_SIZE = 5

RATE_LIMIT_TIME = 8 #15 # Testing 8 and 15 next @todo!
SEMANTIC_THRESH_LIMIT = 75  # Strict percentile boundary for high-precision chunks
SLEEP_TIME_SEC = 3
SLEEP_TIME_INC = 0.50 

# --- Prompt Formats ---
PROMPT_INSTR = """
[OUTPUT FORMAT REQUIREMENT]
Return a valid JSON object where the keys are the string representations of the chunk IDs, and the values are arrays containing exactly 3 hypothetical clinical questions for that specific chunk. 
Do not include any markdown syntax wraps (no triple backticks). Start your response directly with the opening curly brace.

Expected Response Format Example:
{{
  "0": ["Diagnostic criteria question for chunk 0?", "Treatment dosage question for chunk 0?", "Clinical standard question for chunk 0?"],
  "1": ["Diagnostic criteria question for chunk 1?", "Treatment dosage question for chunk 1?", "Clinical standard question for chunk 1?"]
}}
""".strip()

PROMPT_QUESTION_GENERATOR = """
You are an AI {AI_ROLE} specialized in generating precise, clinically relevant questions for information retrieval.
Your task is to analyze the provided text context containing a batch of distinct, tagged chunks, and generate exactly 3 hypothetical questions for each chunk block.

[QUESTION STYLE REQUIREMENTS]
1. Questions must target specific numeric values, treatment thresholds, or clinical facts provided inside that individual chunk ID.
2. Ground clinical terminology strictly in the reference provided. Do not extrapolate beyond the text.

[INPUT DATA]
COMPACTED TEXT CHUNKS:
{docs}

{PROMPT_INSTR}
""".strip()

PROMPT_TABLE_QUESTION_GENERATOR = """
You are an AI {AI_ROLE} specialized in generating precise, clinically relevant questions for information retrieval.
Your task is to analyze the provided text context and table data, and generate exactly 3 hypothetical questions that can be answered definitively by the numeric values or facts inside the table structure.

[INPUT DATA]
--- TEXT ADJACENT TO TABLE ---
{docs}

--- TABLE DATA ---
{tables}

[CRITICAL INSTRUCTIONS]
1. Target specific numbers, metrics, ranges, values, or criteria listed in the table data.
2. Use the adjacent text only to ground clinical terminology.
3. You must output a valid JSON object matching the exact format shown in the example below. Do not deviate from this schema.

[EXPECTED OUTPUT FORMAT EXAMPLE]
{{
  "questions": [
    "What is the recommended serum target level for patient subgroup X?",
    "Which diagnostic criteria metric indicates a severe deficiency threshold?",
    "What specific baseline value must be reached before initiating treatment?"
  ]
}}

[OUTPUT BOUNDARY]
Return ONLY the raw JSON structure. No preamble, no conversational text, no markdown block syntax (do not use triple backticks). Start your response directly with the opening curly brace.
""".strip()

# --- Self Retrievers --- #

# Used for Question Generator
RETRIEVER_DOCUMENT_CONTENT = f"Text Semantic Chunks for {DOCUMENT_FILE} published by the Global Nutritional Health Organization"
RETRIEVER_DOCUMENT_METADATA_FIELDS = [
    AttributeInfo(
        name="source",
        description=f"The exact file path name of the medical reference text (e.g., '{DOCUMENT_FILE}'). Use strictly for matching string values.",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The physical page number within the PDF document. Must use strictly integer values for numerical comparison operations.",
        type="integer",
    ),
    AttributeInfo(
        name="doc_type",
        description="The explicit data classification marker specifying if a chunk is 'source_text', 'hypothetical_questions', or 'table_hypothetical_questions'.",
        type="string",
    )
]

# Used for Table Question Generator
RETRIEVER_HYPER_DOCUMENT_CONTENT = f"Hypothetical clinical research queries derived from the reference workbook: {DOCUMENT_FILE}"
RETRIEVER_HYPER_DOCUMENT_METADATA_FIELDS = [
    AttributeInfo(
        name="original_content",
        description="The source text segment that anchors the generation. Use strictly for text search matching parameters.",
        type="string"
    ),
    AttributeInfo(
        name="source",
        description=f"The file path name string (e.g., '{DOCUMENT_FILEPATH}').",
        type="string"
    ),
    AttributeInfo(
        name="page",
        description="The integer index page number of the medical document text block.",
        type="integer"
    ),
    AttributeInfo(
        name="doc_type",  # 👈 Standardized 'name' and corrected from 'type' to match the database logic character-for-character
        description="The explicit data classification marker specifying if a chunk is 'hypothetical_questions' or 'table_hypothetical_questions'.",
        type="string"
    )
]

RETRIEVER_QUERIES = [
    "What is the recommended dosage for treating scurvy?",
    "What is the definition of Vitamin C, based only on content from pages 1 through 14?",
    f"Describe the clinical signs of nutritional deficiency documented in `{DOCUMENT_FILEPATH}`.",
    "What type of nutritional support is needed for patients to increase lean body mass?",
    "On page 35, please explain the correlation between Vitamin B12 levels and tissue deficiency.",
    "What are the laboratory and clinical standards for diagnosing Vitamin D deficiency in adults?",
]
