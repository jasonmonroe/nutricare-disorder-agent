# main.py

from src.utils import start_timer, show_timer

def run_main_pipeline():
    pass

# --- DECLARE NUTRITION DISORDER AGENT
def nutrition_disorder_streamlit():
    """
    A Streamlit-based UI for the Nutrition Disorder Specialist Agent.
    """
    st.title(f"{TITLE}")
    st.markdown("<hr style='margin: 0'>", unsafe_allow_html=True)
    st.info(body="""
    Welcome! I'm your **Dedicated AI Nutrition Agent**.
    I specialize in providing information about **nutrition disorders**, including **symptoms, causes, treatment options, and preventative measures.**
    I'm ready to answer your health-related questions.
    """, icon="📢")

    st.warning(body=f"Type **{EXIT_CMD}** at anytime to end the conversation.", icon="🪬") # Used EXIT_CMD constant here

    # Initialize the session state for chat history and user_id if they don't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    # Login form: Only if the user is not logged in
    if st.session_state.user_id is None:
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
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input with custom placeholder text.  The user-facing prompt
        user_query = st.chat_input(f"Agent: Ask your question here, {st.session_state.user_id} (or '{EXIT_CMD}')...")

        if user_query:
            if user_query.lower() == EXIT_CMD:
                st.session_state.chat_history.append({"role": "user", "content": EXIT_CMD})

                with st.chat_message("User"):
                    st.write(EXIT_CMD)

                goodbye_msg = "Agent: Goodbye! Feel free to return if you have more questions about nutrition disorders."
                st.session_state.chat_history.append({"role": "assistant", "content": goodbye_msg})

                with st.chat_message("assistant"):
                    st.write(goodbye_msg)

                st.session_state.user_id = None
                st.rerun()
                return

            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("User"):
                st.write(f"{st.session_state.user_id}: {user_query}")

            thinking = st.empty()
            thinking.info(body="Thinking. . .", icon="🤔")

            # Filter input using Llama Guard
            filtered_result = filter_input_with_llama_guard(user_query)
            filtered_result = filtered_result.replace("\n", " ")  # Normalize the result

            # Check if input is safe based on allowed statuses
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
                    error_msg = f"Sorry, I encountered an error while processing your query. Please try again."
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

            thinking.empty()


if __name__ == 'main':
    # Start Program
    run_main_pipeline()
    # End Program

    # --- RUN THE AI AGENT --- #
    start_time = start_timer()
    nutrition_disorder_streamlit()
    show_timer(start_time)







