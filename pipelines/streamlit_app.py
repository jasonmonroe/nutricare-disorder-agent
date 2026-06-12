# src/streamlit.py

import os
import logging
import zipfile
import streamlit as st
from models import ChromaModel

# Local Libraries
from models.agentic_rag_tool import AgenticRagTool
from models.nutrition_bot import NutritionBot
from models.openai import OpenAIModel

from src.config import (
    AGENT_EXIT_CMDS,
    AI_TITLE,
    APP_TITLE,
    DOCUMENT_DIR,
    DOCUMENT_ZIP,
    GROQ_API_KEY,
    I_ANGRY,
    I_BOT,
    I_CROSSMARK,
    I_DOCUMENT,
    I_FLAG,
    I_FROWN,
    I_GEAR,
    I_SAD,
    I_SMILING,
    I_THINKING,
    I_WARNING,
    I_WATCH,
    HF_TOKEN,
    HF_REPO_ID,
    LLAMA_KEY,
    LLAMA_MODEL,
    MEM0_API_KEY,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
)
from src.utils import show_datetime
from tools.agentic_rag import make_agentic_rag_tool

logger = logging.getLogger(__name__)

def get_nutrition_bot_params() -> tuple:
    openai_model = OpenAIModel()
    llm = openai_model.llm
    llm_chatbot = openai_model.llm_chatbot

    # Create vector storage for nutritional information
    chroma_db = ChromaModel({
        'llm': llm,
        'embedding_model': openai_model.embedding_model,
        'collection_name': 'nutritional',
    })

    retriever = chroma_db.retriever
    agentic_rag_tool = AgenticRagTool(llm, retriever)
    workflow_app = agentic_rag_tool.compile()
    rag_tool = make_agentic_rag_tool(llm, retriever, workflow_app)

    return llm_chatbot, rag_tool


@st.cache_resource
def get_chatbot_instance() -> NutritionBot:
    """Initializes and caches the NutritionBot instance."""
    logger.info("\n# --- Loading NutritionBot Unified Instance --- #")
    llm_chatbot, rag_tool = get_nutrition_bot_params()

    return NutritionBot(llm_chatbot, tools=[rag_tool])


class StreamLitApp:

    def __init__(self, llama) -> None:
        self.llama = llama
        self.start_session()

    def start_session(self) -> None:
        # --- INITIALIZE PERSISTENT STATE USING EXPLICIT STRINGS ---
        if "keys_valid" not in st.session_state:
            st.session_state["keys_valid"] = None

        if "doc_found" not in st.session_state:
            st.session_state["doc_found"] = None

        # --- VALIDATE API CREDENTIALS KEYS ---
        if st.session_state["keys_valid"] is None:
            is_valid = self.check_program_keys()
            st.session_state["keys_valid"] = is_valid

        if st.session_state["keys_valid"] is False:
            st.stop()

        # --- FIND & REFERENCE DOCUMENT FOR CHUNKING ---
        if st.session_state["doc_found"] is None:
            doc_found = self.check_document_file()
            st.session_state["doc_found"] = doc_found

        if st.session_state["doc_found"] is False:
            st.stop()

        # Cache the heavy engine interface tool early in the lifecycle
        if "chatbot" not in st.session_state:
            st.session_state["chatbot"] = get_chatbot_instance()

    def check_program_keys(self) -> bool:
        keys_to_check = {
            "GROQ_API_KEY": GROQ_API_KEY,
            "HF_TOKEN": HF_TOKEN,
            "HF_REPO_ID": HF_REPO_ID,
            "LLAMA_KEY": LLAMA_KEY,
            "LLAMA_MODEL": LLAMA_MODEL,
            "MEM0_API_KEY": MEM0_API_KEY,
            "OPENAI_API_KEY": OPENAI_API_KEY,
            "OPENAI_API_BASE": OPENAI_API_BASE,
            "OPENAI_EMBEDDING_MODEL": OPENAI_EMBEDDING_MODEL,
            "OPENAI_MODEL": OPENAI_MODEL
        }

        missing_keys = []
        for key_name, key_value in keys_to_check.items():
            if not key_value:
                missing_keys.append(key_name)
                error_msg = f"{I_CROSSMARK} {key_name} value is None!"
                st.error(error_msg)
                logger.error(error_msg)

        if missing_keys:
            fatal_msg = f"{I_FLAG} FATAL ERROR: Missing secrets... {', '.join(missing_keys)}."
            st.error(fatal_msg)
            logger.critical(fatal_msg)
            return False

        return True

    def check_document_file(self) -> bool:
        if not os.path.isdir(DOCUMENT_DIR):
            warning_msg = f"{I_WARNING} WARNING: Document directory: `{DOCUMENT_DIR}` not found!"
            st.warning(warning_msg)
            logger.warning(warning_msg)

            if not os.path.exists(DOCUMENT_ZIP):
                error_msg = f"{I_FLAG} ERROR: Required zip file `{DOCUMENT_ZIP}` not found. Please upload it."
                st.error(error_msg)
                logger.error(f"Current root tree files: {os.listdir('.')}")
                return False
            else:
                st.info(f"{I_DOCUMENT} Zip file: `{DOCUMENT_ZIP}` found! Extracting...")
                with zipfile.ZipFile(DOCUMENT_ZIP, 'r') as zip_ref:
                    zip_ref.extractall(".")
                st.info(f"{I_DOCUMENT} Zip file: `{DOCUMENT_ZIP}` extracted successfully.")
                return True
        return True

    def show_title(self) -> None:
        st.title(f"{I_BOT} {AI_TITLE}")
        st.markdown("<hr style='margin: 0'>", unsafe_allow_html=True)
        st.info(body=f"""
        Welcome! I'm your **{I_BOT}{APP_TITLE}**.
        I specialize in providing information about **nutrition disorders**, including **symptoms, causes, treatment options, and preventative measures.**
        """.strip(), icon="📢")
        st.warning(body=f"Type **{', '.join(AGENT_EXIT_CMDS)}** at anytime to end the conversation.", icon="🪬")

    def run(self) -> None:
        self.show_title()

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'user_id' not in st.session_state:
            st.session_state.user_id = None

        if st.session_state.user_id is None:
            self._unknown_user()
        else:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # Resolved quotes nesting mismatch on the input placeholder string layout
            exit_options_str = ", ".join(AGENT_EXIT_CMDS)
            user_query = st.chat_input(f"{I_THINKING} Agent: Ask your question here, {st.session_state.user_id} (or '{exit_options_str}')...")

            if user_query:
                if user_query.lower() in AGENT_EXIT_CMDS:
                    self._exit_app()

                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("User"):
                    st.write(f"{st.session_state.user_id}: {user_query}")

                thinking = st.empty()
                thinking.info(body="Thinking. . .", icon=f"{I_THINKING}")

                filtered_result = self.llama.filter_input_with_llama_guard(user_query)
                filtered_result = filtered_result.replace("\n", " ").strip()

                self._handle_input(filtered_result, user_query)
                thinking.empty()

    def _unknown_user(self) -> None:
        with st.form("login_form", clear_on_submit=True):
            st.write(f"{I_WATCH} Session Start: {show_datetime()}")
            user_id = st.text_input("Agent: Please enter your name to begin:").strip()

            if user_id in AGENT_EXIT_CMDS:
                st.error(body="You cannot name yourself a system keyword.", icon="🚨")
                st.stop()

            submit_button = st.form_submit_button("Login")
            if submit_button and user_id:
                st.session_state.user_id = user_id
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"{I_SMILING} Agent: Welcome, {user_id}! How can I help you today?"
                })
                st.session_state.login_submitted = True

        if st.session_state.get("login_submitted", False):
            st.session_state.pop("login_submitted")
            st.rerun()

    def _handle_input(self, filtered_result, user_query) -> None:
        if filtered_result in ["SAFE", "BYPASS_SAFE", ""]:
            try:
                # Reliably pulls from the pre-cached session model safely
                response = st.session_state.chatbot.handle_customer_query(
                    st.session_state.user_id,
                    user_query
                )
                with st.chat_message("assistant"):
                    st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = "Sorry, I encountered an error while processing your query. Please try again."
                error_str = f"Error: {str(e)}"
                with st.chat_message("assistant"):
                    st.error(body=error_str, icon=f"{I_FROWN}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"{error_msg} {error_str}"})
        else:
            inappropriate_msg = "I apologize, but I cannot process that input as it may be inappropriate. Please try again."
            with st.chat_message("assistant"):
                st.warning(body=inappropriate_msg, icon=f"{I_ANGRY}")
            st.session_state.chat_history.append({"role": "assistant", "content": inappropriate_msg})

    def _exit_app(self) -> None:
        exit_cmd_str = ", ".join(AGENT_EXIT_CMDS)
        st.session_state.chat_history.append({"role": "user", "content": exit_cmd_str})

        with st.chat_message("User"):
            st.write(exit_cmd_str)

        goodbye_msg = f"{I_SAD} Agent: Goodbye! Feel free to return if you have more questions about nutrition disorders."
        st.session_state.chat_history.append({"role": "assistant", "content": goodbye_msg})

        with st.chat_message("assistant"):
            st.write(goodbye_msg)

        st.session_state.user_id = None
        st.rerun()
