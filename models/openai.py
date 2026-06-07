# models/openai.py

# https://openai.com
# Documentation: https://developers.openai.com/api/docs

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Local fallback for embeddings when OpenAI auth fails
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
from src.config import (
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_EMB_MODEL,
    OPENAI_MODEL
)


class OpenAIModel:
    def __init__(self):
        self.embedding_model = self._get_embedding_model()
        self.llm = self._load_llm()
        self.llm_chatbot = self._load_llm_chatbot()

    def _get_embedding_model(self) -> OpenAIEmbeddings:

        # Initialize the OpenAI Embeddings
        # see: https://docs.langchain.com/oss/python/integrations/text_embedding/openai
        try:
            emb = OpenAIEmbeddings(
                openai_api_base=OPENAI_API_BASE, # Fill in the endpoint
                openai_api_key=OPENAI_API_KEY,   # Fill in the API key
                model=OPENAI_EMB_MODEL,          # Fill in the model name
                max_retries=8,                   # openai client retries, Added for robustness (was =3)
                request_timeout=60,              # avoid timeouts on backoff
            )

            # Quick smoke-test to ensure credentials are valid (small batch)
            try:
                emb.embed_documents(["test"])
            except Exception:
                raise

            return emb
        except Exception:
            # Fallback to local SentenceTransformer embeddings to allow offline/test runs
            if SentenceTransformer is None:
                raise RuntimeError("OpenAI embeddings unavailable and SentenceTransformer is not installed.")

            class LocalSentenceEmbeddings:
                def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
                    self.model = SentenceTransformer(model_name)

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    arr = self.model.encode(texts, show_progress_bar=False)
                    # Ensure result is a list of lists of plain Python floats
                    return [[float(v) for v in x] for x in arr]

            print("[INFO] OpenAI embeddings unavailable — falling back to local SentenceTransformer embeddings.")
            return LocalSentenceEmbeddings()

    def _load_llm(self) -> ChatOpenAI:
        # This initializes the OpenAI embeddings model using the specified endpoint, API key, and model name.
        # This initializes the Chat OpenAI model using the provided endpoint, API key, deployment name.

        # Initialize the Chat OpenAI model
        return ChatOpenAI(
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

    def _load_llm_chatbot(self) -> ChatOpenAI:
        # Note: This is for Nutrition Bot
        return ChatOpenAI(
            model_name=OPENAI_MODEL,  # Specify the model to use (e.g., a GPT-4 optimized version)
            openai_api_key=OPENAI_API_KEY,  # API key for authentication
            base_url = OPENAI_API_BASE,
            temperature=0  # Controls randomness in responses; 0 ensures deterministic results
        )

    @staticmethod
    def filter_response(self, resp: str, index=None) -> str:
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
