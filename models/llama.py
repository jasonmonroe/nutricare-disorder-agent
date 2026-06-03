# models/llama.py


# --- FILTER INPUT WITH LLAMA GUARD
# Initialize the Llama Guard client with the API key
llama_guard_client = Groq(api_key=GROQ_API_KEY)

# Set the LLM and embedding model in the LlamaIndex settings.
Settings.llm = llm
Settings.embedding = embedding_model


# Initialize LlamaParse with desired settings
parser = LlamaParse(
    result_type="markdown",  # Specify the result format
    skip_diagonal_text=True, # Skip diagonal text in the PDFs
    fast_mode=False,         # Use normal mode for parsing
    num_workers=9,           # Number of workers for parallel processing
    check_interval=10,       # Check interval for processing
    api_key=LLAMA_KEY        # API key for LlamaParse
)

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