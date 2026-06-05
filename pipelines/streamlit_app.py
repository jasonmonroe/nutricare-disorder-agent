# src/streamlit.py

# https://streamlit.io
# Documentation: https://docs.streamlit.io

# Python Libraries
import os
import zipfile

# Vendor Libraries
import streamlit as st

# Local Libraries
from models.nutrition_bot import NutritionBot
from src.config import (
    AI_TITLE,
    EXIT_CMD,
    HF_TOKEN,
    GROQ_API_KEY,
    LLAMA_KEY,
    MEM0_API_KEY,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    DOCUMENT_DIR,
    DOCUMENT_ZIP, APP_TITLE
)
from src.utils import show_datetime



# Cache ChatBot Instance
@st.cache_resource
def get_chatbot_instance() -> NutritionBot:
    """
    Initializes and caches the NutritionBot instance.

    :return:
    """
    print("# --- Loading NutritionBot --- #")

    return NutritionBot()


class StreamLitApp():
    def __init__(self, llama) -> None:

        self.llama = llama

        self.start_session()
        self.check_program_keys()
        self.check_document_file()


    def start_session(self) -> None:
        # --- INITIALIZE PERSISTENT STATE ---
        session_keys_valid = None
        session_doc_found = None

        if session_keys_valid not in st.session_state:
            st.session_state[session_keys_valid] = session_keys_valid

        if session_doc_found not in st.session_state:
            st.session_state[session_doc_found] = session_doc_found

        # --- VALIDATE API CREDENTIALS KEYS AND CHECK THE SOURCE FILE
        if st.session_state[session_keys_valid] is None:
            is_valid = self.check_program_keys()
            st.session_state[session_keys_valid] = is_valid

            if not is_valid:
                st.stop()

        # --- FIND & REFERENCE DOCUMENT FOR CHUNKING
        if st.session_state[session_doc_found] is None:
            doc_found = self.check_document_file()
            st.session_state[session_doc_found] = doc_found

            if not doc_found:
                st.stop()

    # Checks if all necessary keys are being used
    def check_program_keys(self) -> bool:
        # Load keys and check if any or missing to kill the script.
        keys_to_check = {
            "HF_TOKEN": HF_TOKEN,
            "GROQ_API_KEY": GROQ_API_KEY,
            "LLAMA_KEY": LLAMA_KEY, # This is the alias for os.getenv("LLAMA_KEY")
            "MEM0_API_KEY": MEM0_API_KEY,
            "OPENAI_API_KEY": OPENAI_API_KEY, # formerly config.json("API_KEY")
            "OPENAI_API_BASE": OPENAI_API_BASE,
        }

        missing_keys = []
        for key_name, key_value in keys_to_check.items():
            error_msg = f"{key_name} value is None!"
            if not key_value: # Checks if the value is None (i.e., not found)
                missing_keys.append(key_name)
            if key_value is None:
                st.error(error_msg)
                print(error_msg)

        error_msg = f"FATAL ERROR: The following secrets are missing... {', '.join(missing_keys)}."
        if missing_keys:
            st.error(error_msg)
            print(error_msg)
            return False

        return True


    # @todo - should this helper file be in src/utils.py or in the streamlist class?
    # Checks if the document directory exists
    def check_document_file(self) -> bool:

        if not os.path.isdir(DOCUMENT_DIR):
            st.warning(f"WARNING: Document directory: `{DOCUMENT_DIR}`  not found!")
            print(f"WARNING: Document directory: `{DOCUMENT_DIR}`  not found!")

            # Check for a zip file
            if not os.path.exists(DOCUMENT_ZIP):
                st.error(f"ERROR: Required zip file `{DOCUMENT_ZIP}` not found either.  Please upload it.")
                print(os.listdir('.'))
                return False

            else:
                st.info(f"Zip file: `{DOCUMENT_ZIP}` found!\nExtracting zip file...")
                print(f"Zip file: `{DOCUMENT_ZIP}` found!\nExtracting zip file...")

                with zipfile.ZipFile(DOCUMENT_ZIP, 'r') as zip_ref:
                    zip_ref.extractall(".")

                st.info(f"Zip file: `{DOCUMENT_ZIP}` extracted.")
                print(f"Zip file: `{DOCUMENT_ZIP}` extracted.")
                return True

        else:
            print(f"Document directory found: `{DOCUMENT_DIR}`.")
            return True


    def show_title(self) -> None:
        st.title(f"{AI_TITLE}")
        st.markdown("<hr style='margin: 0'>", unsafe_allow_html=True)
        st.info(body=f"""
        Welcome! I'm your **{APP_TITLE}**.
        I specialize in providing information about **nutrition disorders**, including **symptoms, causes, treatment options, and preventative measures.**
        I'm ready to answer your health-related questions.
        """, icon="📢")

        st.warning(body=f"Type **{EXIT_CMD}** at anytime to end the conversation.", icon="🪬") # Used EXIT_CMD constant here


    def run(self) -> None:
        """
        A Streamlit-based UI for the Nutrition Disorder Specialist Agent.
        """
        self.show_title()

        # Initialize the session state for chat history and user_id if they don't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'user_id' not in st.session_state:
            st.session_state.user_id = None

        # Login form: Only if the user is not logged in
        if st.session_state.user_id is None:
            self._unknown_user()

        else:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # Chat input with custom placeholder text.  The user-facing prompt
            user_query = st.chat_input(f"Agent: Ask your question here, {st.session_state.user_id} (or '{EXIT_CMD}')...")

            if user_query:
                if user_query.lower() == EXIT_CMD:
                    self._exit_app()

                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("User"):
                    st.write(f"{st.session_state.user_id}: {user_query}")

                thinking = st.empty()
                thinking.info(body="Thinking. . .", icon="🤔")

                # Filter input using Llama Guard
                filtered_result = self.llama.filter_input_with_llama_guard(user_query)
                filtered_result = filtered_result.replace("\n", " ")  # Normalize the result

                # Check if input is safe based on allowed statuses
                self._handle_input(filtered_result, user_query)

                thinking.empty()


    def _unknown_user(self) -> None:
        with st.form("login_form", clear_on_submit=True):
            st.write(f"Session Start: {show_datetime()}")
            user_id = st.text_input("Agent: Please enter your name to begin:").strip()

            # Don't let the username themselves a keyword
            if EXIT_CMD in user_id:
                st.error(body="You cannot name yourself a keyword.", icon="🚨")
                st.stop()

            submit_button = st.form_submit_button("Login")

            if submit_button and user_id:
                st.session_state.user_id = user_id
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Agent: Welcome, {user_id}! How can I help you with nutrition disorders today?"
                })
                st.session_state.login_submitted = True  # Set flag to trigger rerun

        if st.session_state.get("login_submitted", False):
            st.session_state.pop("login_submitted")
            st.rerun()


    def _handle_input(self, filtered_result, user_query):
        if filtered_result in ["SAFE", "BYPASS_SAFE", ""]:
            try:

                # Get the cached chatbot instance
                st.session_state.chatbot = get_chatbot_instance()
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
                    st.error(body=error_str, icon="😩")
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg + " " + error_str})

        else:
            # Unsafe queries are handled here!
            inappropriate_msg = "I apologize, but I cannot process that input as it may be inappropriate. Please try again."
            with st.chat_message("assistant"):
                st.warning(body=inappropriate_msg, icon="🤬")

            st.session_state.chat_history.append({"role": "assistant", "content": inappropriate_msg})

    def _exit_app(self) -> None:
        st.session_state.chat_history.append({"role": "user", "content": EXIT_CMD})

        with st.chat_message("User"):
            st.write(EXIT_CMD)

        goodbye_msg = "Agent: Goodbye! Feel free to return if you have more questions about nutrition disorders."
        st.session_state.chat_history.append({"role": "assistant", "content": goodbye_msg})

        with st.chat_message("assistant"):
            st.write(goodbye_msg)

        st.session_state.user_id = None
        st.rerun()
