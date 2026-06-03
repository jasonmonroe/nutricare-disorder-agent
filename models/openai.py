# models/openai.py


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


class OpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_text(self, prompt: str) -> str:
        return self.api_key.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )


        

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