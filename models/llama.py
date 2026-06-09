# models/llama.py

# +---------------+
# |     LLAMA     |
# +---------------+

# Vendor Libraries
from groq import Groq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LlamaParse & LlamaIndex imports
from llama_parse import LlamaParse  # Document parsing library
from llama_index.core import Settings, SimpleDirectoryReader  # Core functionalities of the LlamaIndex

# Local
from src.config import (
    GROQ_API_KEY,
    I_CROSSMARK,
    I_PEN,
    LLAMA_KEY,
    LLAMA_MODEL,
    LLAMA_UNSAFE_CODES
)


class LlamaModel:

    """
    This guide provides information and resources to help you set up Llama including how to access the model, hosting, 
    how-to and integration guides. Additionally, you will find supplemental materials to further assist you while building 
    with Llama.

    https://www.llama.com/docs/overview/
    """

    def __init__(self, llm: ChatOpenAI, embedding_model: OpenAIEmbeddings, log: bool=False):
        """
        Initialize the Llama Guard client with the API key.  Set the LLM and embedding model in the LlamaIndex settings.

        :param log: determines if logs will be outputted in terminal
        :param llm:
        :param embedding_model:
        """

        self._log = log

        self.llama_guard_client = Groq(api_key=GROQ_API_KEY)
        self.parser = self._get_parser()

        Settings.llm = llm
        Settings.embedding = embedding_model

    def _get_parser(self) -> LlamaParse:
        """
        Initialize LlamaParse with desired settings
        :return: LLamaParse
        """

        return LlamaParse(
            result_type='markdown',  # Specify the result format
            skip_diagonal_text=True, # Skip diagonal text in the PDFs
            fast_mode=False,         # Use normal mode for parsing
            num_workers=9,           # Number of workers for parallel processing
            check_interval=10,       # Check interval for processing
            api_key=LLAMA_KEY        # API key for LlamaParse
        )

    def filter_input_with_llama_guard(self, user_input_str: str) -> str:
        """
        Function to filter user input with Llama Guard

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
            llama_response = self.llama_guard_client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": user_input_str.strip()
                }],
                model=LLAMA_MODEL,
            )

            # Return the filtered input
            result = llama_response.choices[0].message.content.strip()

            if self._log:
                print(f"\n# --- {I_PEN}  Open Guard result {I_PEN} --- #")
                print(result)
                print(f"# --- {I_PEN}  Close Guard result {I_PEN} --- #\n")

            if "unsafe" in result:
                if any(code.strip() in LLAMA_UNSAFE_CODES for code in result.replace("unsafe ", "").strip().split(",")):
                    return "BYPASS_SAFE"
                else:
                    return "UNSAFE"
            else:
                return "SAFE"

        except Exception as e:
            print(f"{I_CROSSMARK} Error with Llama Guard: {e}")
            return ""
            