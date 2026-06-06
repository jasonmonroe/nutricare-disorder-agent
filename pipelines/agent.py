# pipelines/agent.py

import nest_asyncio

# Local Libraries
from models.agentic_rag_tool import AgenticRagTool
from models.nutrition_bot import NutritionBot
from tools.agentic_rag import make_agentic_rag_tool

from src.config import AI_TITLE, EXIT_CMD
from src.utils import show_datetime, start_timer, get_time

"""
Section 2: Building an Intelligent Nutrition Disorder Agent with Advanced Retrieval and Safety Mechanisms
"""

def build(dataset: dict):
    """
    Builds the agentic app by compile workflow object.
    :param dataset:
    :param show_logs:
    :return:
    """
    """
    openai_model = OpenAIModel()
    llm = openai_model.load_llm()
    #llama = LlamaModel(llm, openai_model.embedding_model)

    # --- INITIALIZE CHROMA VECTOR STORAGE FOR RETRIEVING DOCUMENTS
    # Retrieve `nutritional` database created from Google Colab

    # Create vector storage for nutritional information
    chroma_db = ChromaModel({
        'llm': llm,
        'embedding_model': openai_model.embedding_model,
        'collection_name': 'nutritional'
    })
    """

    chroma_db = dataset['chroma_db']
    openai_model = dataset['openai_model']
    llm = openai_model.llm


    # Stage 2 - Start Program
    # Apply the nested async loop to allow async code execution in the notebook
    nest_asyncio.apply()

    # --- Visualize Workflow --- #
    agentic_rag_tool = AgenticRagTool(llm, chroma_db.retriever)
    workflow_app = agentic_rag_tool.compile()
    agentic_rag_tool.display_workflow(workflow_app)

    return workflow_app


def start(dataset: dict) -> None:
    """
    Starts the agentic app!

    A conversational agent that answers nutrition-disorder-related questions using a RAG-based workflow with safety
    filtering and user session handling.
    :param dataset:
    :return:
    """

    print(f"""
        +-------------------------------------+
        | {AI_TITLE:^45}|
        +-------------------------------------+
        | Welcome! I'm your dedicated AI Nutrition Agent. |
        | Ask me anything about nutrition disorders. You can inquire about |
        | symptoms, causes, treatment options, or preventative measures. |
        | I'm ready to help with your health-related questions. |
        |
        | Type '{EXIT_CMD}' to end the conversation. |
        +-------------------------------------+
    """)

    # Initialize streamlit persistent state
    show_logs = dataset.get('show_log', False)
    print(f'DEBUG: show_logs:{show_logs}')

    openai_model = dataset['openai_model']
    llm_chatbot = openai_model.llm_chatbot
    llm = openai_model.llm
    llama = dataset['llama']
    chroma_db = dataset['chroma_db']

    rag_tool = make_agentic_rag_tool(llm, chroma_db.retriever)
    chatbot = NutritionBot(llm_chatbot, tools=[rag_tool])
    chatbot.agent_executor.verbose = show_logs  # Set logging preferences

    # This provides a way to initiate a chat as different users.
    user_id = input("Agent: Login by providing customer name ")  # Get user ID for tracking conversation sessions

    print(f"\n--- Session Start: {show_datetime()} ---\n")

    while True:
        # Get user input
        print("Agent: How can I help you?\n")
        user_query = input(f"{user_id}: ")

        # Set timer for each question
        q_time = start_timer()

        # Define the logic for exiting the loop' [if the user types in exit]
        if user_query.lower() == EXIT_CMD:
            print("\nAgent: Goodbye! Feel free to return if you have more questions.")
            print(f"--- Session End: {show_datetime()} ---")
            break

        # Filter input through Llama Guard - returns "SAFE" or "UNSAFE"
        filtered_result = llama.filter_input_with_llama_guard(user_query) # Call function to filter input
        filtered_result = filtered_result.replace("\n", " ")   # Normalize the result

        # Check if filtered_result is SAFE or UNSAFE
        if filtered_result in ["SAFE", "BYPASS_SAFE"]:
            # Process the user query using the RAG workflow
            try:
                response = chatbot.handle_customer_query(user_id, user_query)  # Call chatbot handler function
                print(f"Agent: {response}\n")

            except Exception as e:
                print("Agent: Sorry, I encountered an error while processing your query. Please try again.")
                print(f"Customer Query Error: {e}\n")
        else:
            print(f"Agent: I apologize, but I cannot process that input `{filtered_result}` as it may be inappropriate. Please try again.")

        # Show answer duration per query
        print(f"[Answered in {get_time(q_time)}]\n")
