# pipelines/agent.py

import nest_asyncio
import sys
import time

from models.agentic_rag_tool import AgenticRagTool
from models.nutrition_bot import NutritionBot
from tools.agentic_rag import make_agentic_rag_tool

from src.config import (
    I_CLOCK,
    I_CROSSMARK,
    I_RUNNING,
    I_SAD,
    I_SMILING,
    I_STAR,
    I_THINKING,
    I_WARNING,
    I_WATCH,
    LLAMA_SAFE,
)
from src.utils import show_ai_agent_banner, show_datetime, start_timer, get_time


def build(dataset: dict):
    print(f'\n# --- {I_RUNNING} Start Building agent pipeline {I_RUNNING} --- #')
    chroma_db = dataset['chroma_db']
    openai_model = dataset['openai_model']
    llm = openai_model.llm

    nest_asyncio.apply()

    agentic_rag_tool = AgenticRagTool(llm, chroma_db.retriever)
    workflow_app = agentic_rag_tool.compile()
    agentic_rag_tool.display_workflow(workflow_app)

    print(f'\n# --- {I_RUNNING} Completed agent pipeline {I_RUNNING} --- #')
    return workflow_app


def start(dataset: dict) -> None:
    print(f'\n# --- {I_RUNNING} Starting agent pipeline {I_RUNNING} --- #')
    show_ai_agent_banner()

    show_logs = dataset.get('log', False)
    chroma_db = dataset.get('chroma_db', None)
    openai_model = dataset.get('openai_model', None)
    llama = dataset.get('llama', None)
    workflow_app = dataset.get('workflow_app', None)

    # Safeguard block evaluation order prevents unhandled NoneType errors
    if chroma_db is None or openai_model is None or llama is None:
        print(f"{I_CROSSMARK} Core dependencies didn't load properly. Exiting system!!! {I_CROSSMARK}")
        sys.exit(0)

    if chroma_db.get_document_count() == 0:
        print(f"{I_CROSSMARK} No documents found in the vector store. Please run with --data first! {I_CROSSMARK}")
        raise RuntimeError("Vector database is completely empty!\n")

    # --- Run --- #

    llm = openai_model.llm
    llm_chatbot = openai_model.llm_chatbot

    nest_asyncio.apply()

    rag_tool = make_agentic_rag_tool(llm, chroma_db.retriever, workflow_app, dataset['log'])
    chatbot = NutritionBot(llm_chatbot, tools=[rag_tool])
    chatbot.agent_executor.verbose = show_logs
    chatbot.start_session()

    # Get user ID for tracking conversation sessions
    user_id = input(f"\n{I_THINKING} Agent: Tell me, what is your name? _ ").strip()
    if not user_id:
        user_id = "User"

    print(f"\n# --- Session Start: {I_CLOCK} {show_datetime()} --- #\n")

    while True:
        if chatbot.has_session_exp():
            print(f'{I_WARNING} Session has expired.  Exiting chat.')
            break

        print(f"{I_SMILING} Agent: How can I help you?\n")
        user_query = input(f"{I_STAR} {user_id}: ")

        # Update input timestamp using absolute standard time epoch
        q_start = time.time()
        chatbot.update_latest_input_at(q_start)

        action = chatbot.check_user_input(user_query)
        if action == 'break':
            break
        if action == 'continue':
            continue

        # Filter input through Llama Guard
        filtered_result = llama.filter_input_with_llama_guard(user_query)

        if filtered_result in LLAMA_SAFE:
            try:
                response = chatbot.handle_customer_query(user_id, user_query)
                print(f"\n{I_SMILING} Agent: {response}\n")
            except Exception as e:
                print(f"\n{I_SAD} Agent: Sorry, I encountered an error while processing your query.")
                print(f"{I_CROSSMARK} Customer Query Error: {e}\n")
        else:
            print(f"\n{I_SAD} Agent: I apologize, but I cannot process that input as it may be inappropriate.")

        print(f"[{I_WATCH} Answered in {get_time(q_start)}]\n")

    # --- Outside of loop --- #
    print(f'{I_WATCH} Session Duration: {chatbot.get_session_duration(start_timer())}')
