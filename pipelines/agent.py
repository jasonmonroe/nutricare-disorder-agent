# pipelines/agent.py

# +---------------+
# |     AGENT     |
# +---------------+

# Python Libraries
from typing import Any
from langgraph.graph.state import CompiledStateGraph
import nest_asyncio

# Local Libraries
from models.agentic_rag_tool import AgenticRagTool
from models.nutrition_bot import NutritionBot
from tools.agentic_rag import make_agentic_rag_tool

from src.config import (
    EXIT_CMD, 
    I_CLOCK, 
    I_CONFUSED, 
    I_CROSSMARK, 
    I_RUNNING, 
    I_SAD, 
    I_SMILING, 
    I_SURPRISED, 
    I_THINKING, 
    I_WATCH
)
from src.utils import show_ai_agent_banner, show_datetime, start_timer, get_time

"""
Section 2: Building an Intelligent Nutrition Disorder Agent with Advanced Retrieval and Safety Mechanisms
"""

def build(dataset: dict) -> CompiledStateGraph:
    """
    Builds the agentic app by compile workflow object.
    :param dataset:
    :return: CompiledStateGraph
    """
    print(f'\n# --- {I_RUNNING} Start Building agent pipeline {I_RUNNING} --- #')

    chroma_db = dataset['chroma_db']
    openai_model = dataset['openai_model']
    llm = openai_model.llm

    # Apply the nested async loop to allow async code execution in the notebook
    nest_asyncio.apply()

    # --- Visualize Workflow --- #
    agentic_rag_tool = AgenticRagTool(llm, chroma_db.retriever)
    workflow_app = agentic_rag_tool.compile()
    agentic_rag_tool.display_workflow(workflow_app)

    print(f'\n# --- {I_RUNNING} Completed agent pipeline {I_RUNNING} --- #')

    return workflow_app

def start(dataset: dict) -> None:
    print(f'\n# --- {I_RUNNING} Starting agent pipeline {I_RUNNING} --- #')

    """
    Starts the agentic app!

    A conversational agent that answers nutrition-disorder-related questions using a RAG-based workflow with safety
    filtering and user session handling.

    :param dataset: dict
    :return: None
    """

    show_ai_agent_banner()

    # Initialize streamlit persistent state
    show_logs = dataset.get('log', False)
    print(f'DEBUG: show_logs:{show_logs}')

    chroma_db = dataset.get('chroma_db', None)
    openai_model = dataset.get('openai_model', None)
    llama = dataset.get('llama', None)
    workflow_app = dataset.get('workflow_app', None) 

    llm = openai_model.llm
    llm_chatbot = openai_model.llm_chatbot

    # Apply the nested async loop to allow async code execution in the notebook
    nest_asyncio.apply()
    
    rag_tool = make_agentic_rag_tool(llm, chroma_db.retriever, workflow_app) 
    chatbot = NutritionBot(llm_chatbot, tools=[rag_tool])
    chatbot.agent_executor.verbose = show_logs  # Set logging preferences
    chatbot.start_session()
    q_time = 0

    user_id = input(f"{I_THINKING} Agent: Tell me, what is your name? _ ")  # Get user ID for tracking conversation sessions
    print(f"\n# --- Session Start: {I_CLOCK} {show_datetime()} --- #\n")

    while chatbot.has_session_exp():
        
        # Get user input
        print(f"{I_SMILING} Agent: How can I help you?\n")
        user_query = input(f"{user_id}: ")

        # Set timer for each question
        q_time = start_timer()
        chatbot.update_latest_input_at(q_time)

        # Define the logic for exiting the loop' [if the user types in exit]
        if user_query.lower() == EXIT_CMD:
            print(f"\n{I_SURPRISED} Agent: Goodbye! Feel free to return if you have more questions.")
            print(f"# --- Session End: {I_CLOCK} {show_datetime()} --- #")
            q_time = start_timer()
            break

        # Note: If user just enters blank, skip Llama and ask for another query.
        if user_query == '':
            print(f'{I_CONFUSED} You did\'t say anything {user_id}.  What\'s your question?')
            continue

        # Filter input through Llama Guard - returns "SAFE" or "UNSAFE"
        filtered_result = llama.filter_input_with_llama_guard(user_query) # Call function to filter input
        filtered_result = filtered_result.replace("\n", " ").strip()   # Normalize the result

        # Check if filtered_result is SAFE or UNSAFE
        if filtered_result in ["SAFE", "BYPASS_SAFE"]:
            # Process the user query using the RAG workflow.
            try:
                response = chatbot.handle_customer_query(user_id, user_query)  # Call chatbot handler function
                print(f"{I_SMILING} Agent: {response}\n")

            except Exception as e:
                print(f"{I_SAD} Agent: Sorry, I encountered an error while processing your query. Please try again.")
                print(f"{I_CROSSMARK} Customer Query Error: {e}\n")
        else:
            print(f"{I_SAD} Agent: I apologize, but I cannot process that input `{filtered_result}` as it may be inappropriate. Please try again.")

        # Show answer duration per query
        print(f"[{I_WATCH} Answered in {get_time(q_time)}]\n")


    # Display session duration
    print(f'{I_WATCH} Session Duration: {chatbot.get_session_duration(q_time)}')
