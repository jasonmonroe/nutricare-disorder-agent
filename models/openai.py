from __future__ import annotations
# models/openai.py

# Vendor Libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Local Libraries
from src.constants import (
    AGENT_EMPTY_RESP,
    I_FLAG,
    I_WARNING,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
)
from src.model_config import ModelConfig

class OpenAIModel:
    # https://openai.com
    # Documentation: https://developers.openai.com/api/docs

    def __init__(self):

        self.embedding_model = self._get_embedding_model() if ModelConfig.is_premium() else self._get_hf_embedding_model()
        self.llm = self._load_llm()
        self.llm_chatbot = self._load_llm_chatbot()

    def _get_embedding_model(self) -> OpenAIEmbeddings:
        """
        We're not using this at this time for this project on a free tier!
        Get the embedding model. Was used with `text-embedding-3-small`
        Uses HuggingFaceEmbeddings for local processing
        which avoids rate limits and API quota issues.

        ℹ️ Note: This function is used for premium models.
        """

        # Fallback to OpenAI embeddings (if using actual OpenAI)
        return OpenAIEmbeddings(
            openai_api_base=OPENAI_API_BASE, # Fill in the endpoint
            openai_api_key=OPENAI_API_KEY,   # Fill in the API key
            model=OPENAI_EMBEDDING_MODEL,          # Fill in the model name
            max_retries=2,                   # openai client retries, Added for robustness (was =3)
            request_timeout=60,              # avoid timeouts on backoff
        )

    def _get_hf_embedding_model(self) -> HuggingFaceEmbeddings:
    
        """
        Instantiate a local embedding layer that bypasses OpenAI / Groq network proxy formatting entirely

        Optimization: Use 'mps' for Mac GPU acceleration, fallback to 'cpu'
        model_kwargs: Arguments passed directly to the base underlying class constructor when downloading, initializing, or loading model files (e.g., target device, trust_remote_code, or file loading preferences like local_files_only).
        encode_kwargs: Arguments passed to the model's forward execution pass during runtime when transforming a text string into an array matrix

        ℹ️ Note: We're using this for the free tier models!
        """

        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        return HuggingFaceEmbeddings(
            model_name=OPENAI_EMBEDDING_MODEL,
            model_kwargs={
                'device': device,
                'local_files_only': True # Free tier model
                },
            encode_kwargs={
                'normalize_embeddings': True  # (Optional: Example of a valid encoding argument)
            }
        )

    def _load_llm(self) -> ChatOpenAI:
        # This initializes the Chat OpenAI model using the provided endpoint, API key, deployment name.

        return ChatOpenAI(
            openai_api_base=OPENAI_API_BASE, # Fill in the endpoint
            openai_api_key=OPENAI_API_KEY,   # Fill in the API key
            max_tokens=None,
            max_retries=2,                   # Retry failed calls
            model=OPENAI_MODEL,              # Fill in the deployment name (e.g., gpt-4o-mini)
            streaming=False,
            temperature=0.0,                 # Set for factual, deterministic output for robustness and quality
        )

    def _load_llm_chatbot(self) -> ChatOpenAI:
        # ℹ️ Note: This is for Nutrition Bot
        return ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_base=OPENAI_API_BASE,
            openai_api_key=OPENAI_API_KEY,  # API key for authentication
            temperature=0  # Controls randomness in responses; 0 ensures deterministic results
        )

    @staticmethod
    def filter_response(resp: str, index=None) -> str:
        # Check if the response is already a string (raw output)
        if isinstance(resp, str):
            content = resp.strip()

        # Check if the response is a LangChain Message object
        elif hasattr(resp, 'content'):
            content = resp.content.strip()

        # Handle unexpected types
        else:
            print(f'{I_WARNING} Warning: Unexpected response type for chunk {index}. Type: {type(resp)}')
            return AGENT_EMPTY_RESP # Treat unexpected types as an empty response

        if len(content) == 0:
            print(f'{I_FLAG} No generated hypothetical questions found for chunk {index}.')
            return AGENT_EMPTY_RESP

        # The output is wrapped in outer quotes and parentheses, e.g., ("['...']")
        # Check for and remove the outer parentheses and quotes if present
        if content.startswith('(') and content.endswith(')'):

            # Remove the outer parentheses
            content = content.strip()[1:-1].strip()

            # Remove the outer quotes that might remain
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]

        # Important: In the LLM-as-string case, you might get "[]" here.
        # The calling code handles the difference between an empty string and the literal string "[]"
        return content
