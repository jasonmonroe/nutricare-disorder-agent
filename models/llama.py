# models/llama.py


# Vendors

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LlamaParse & LlamaIndex imports
from llama_parse import LlamaParse  # Document parsing library
from llama_index.core import Settings, SimpleDirectoryReader  # Core functionalities of the LlamaIndex
from groq import Groq

# Local

from src.config import GROQ_API_KEY, LLAMA_KEY, LLAMA_MODEL, LLAMA_UNSAFE_CODES


class LlamaModel():
    def __init__(self, llm: ChatOpenAI, embedding_model: OpenAIEmbeddings):
        self.llama_guard_client = Groq(api_key=GROQ_API_KEY)
        self.parser = self._get_parser()


        # --- FILTER INPUT WITH LLAMA GUARD
        # Initialize the Llama Guard client with the API key


        # Set the LLM and embedding model in the LlamaIndex settings.
        Settings.llm = llm
        Settings.embedding = embedding_model
        

        
 

    def _get_parser(self):
        # Initialize LlamaParse with desired settings
        return LlamaParse(
            result_type="markdown",  # Specify the result format
            skip_diagonal_text=True, # Skip diagonal text in the PDFs
            fast_mode=False,         # Use normal mode for parsing
            num_workers=9,           # Number of workers for parallel processing
            check_interval=10,       # Check interval for processing
            api_key=LLAMA_KEY        # API key for LlamaParse
        )


    # Function to filter user input with Llama Guard
    def filter_input_with_llama_guard(self, user_input_str: str, model=LLAMA_MODEL) -> str:
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
            llama_response = self.llama_guard_client.chat.completions.create(
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
            