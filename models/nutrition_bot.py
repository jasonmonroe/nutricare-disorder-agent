from __future__ import annotations
# models/nutrition_bot.py

# +-----------------------+
# |     NUTRITION BOT     |
# +-----------------------+

# Python Libraries
import logging
from typing import Any
from datetime import datetime
from mem0 import MemoryClient

# Vendor Libraries
# LangChain imports
from langchain_core.prompts import ChatPromptTemplate as CoreChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# Local Libraries
from src.constants import (
    AGENT_EXIT_CMDS,
    AGENT_RETRIEVAL_LIMIT,
    I_CLOCK,
    I_SURPRISED,
    I_THUMBS_DOWN,
    INACTIVE_SESSION_DUR,
    MEM0_API_KEY,
)
from src.utils import get_time, start_timer, show_datetime


class NutritionBot:
    def __init__(self, llm_chatbot, tools: list):
        f"""
         Initialize the NutritionBot class, setting up memory, the LLM client, tools, and the agent executor.

        :param llm_chatbot:
        :param tools
        """

        self._session_starts_at = None
        self._latest_input_at = None

        # Initialize a memory client to store and retrieve customer interactions
        self.memory = MemoryClient(api_key=MEM0_API_KEY)  # Complete the code to define the memory client API key

        # Initialize the OpenAI client using the provided credentials
        self.client = llm_chatbot

        # Define the system prompt to set the behavior of the chatbot
        system_prompt = """You are a caring and knowledgeable Medical Support Agent, specializing in nutrition disorder-related guidance. Your goal is to provide accurate, empathetic, and tailored nutritional recommendations while ensuring a seamless customer experience.
                          Guidelines for Interaction:
                          Maintain a polite, professional, and reassuring tone.
                          Show genuine empathy for customer concerns and health challenges.
                          Reference past interactions to provide personalized and consistent advice.
                          Engage with the customer by asking about their food preferences, dietary restrictions, and lifestyle before offering recommendations.
                          Ensure consistent and accurate information across conversations.
                          If any detail is unclear or missing, proactively ask for clarification.
                          Always use the agentic_rag tool to retrieve up-to-date and evidence-based nutrition insights.
                          Keep track of ongoing issues and follow-ups to ensure continuity in support.
                          Your primary goal is to help customers make informed nutrition decisions that align with their health conditions and personal preferences.
        """.strip()

        

        # Build the prompt template for the agent
        prompt = CoreChatPromptTemplate.from_messages([
            ("system", system_prompt),  # System instructions
            ("human", "{input}"),  # Placeholder for human input
            ("placeholder", "{agent_scratchpad}")  # Placeholder for intermediate reasoning steps
        ])

        # Create an agent capable of interacting with tools and executing tasks
        agent = create_tool_calling_agent(self.client, tools, prompt)

        # Wrap the agent in an executor to manage tool interactions and execution flow
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def start_session(self):
        self._session_starts_at = start_timer()

    def update_latest_input_at(self, input_at: float) -> None:
        self._latest_input_at = input_at

    def has_session_exp(self) -> bool:
        # If chat session just started return False.  Next time it will be evaluated
        if self._latest_input_at is None:
            return False

        # Max session is 20 minutes.  Anything after that needs to be run again.
        diff_in_secs = abs(start_timer() - self._latest_input_at)

        if diff_in_secs > INACTIVE_SESSION_DUR:
            print(f'\n{I_THUMBS_DOWN} Session has expired.  Exiting chat.')
            return True

        return False

    def get_session_duration(self, session_ends_at) -> str:
        """
        Set duration
        :param session_ends_at:
        :return:
        """
        return get_time(self._session_starts_at, session_ends_at)

    def store_customer_interaction(self, user_id: str, message: str, response: str, metadata: dict) -> None:
        """
        Store customer interaction in memory for future reference.
        """

        metadata["timestamp"] = datetime.now().isoformat()

        conversation = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]

        # Wrap the API call in a try/except guardrail so a memory glitch
        # never crashes the entire user-facing agent session again.
        try:
            self.memory.add(
                conversation,
                user_id=user_id,
                output_format="v1.1",
                metadata=metadata
            )
            print(f"✨ Successfully synchronized memory loop for user: {user_id}")

        except Exception as mem_err:
            # Log the error but let the agent keep running smoothly
            print(f"⚠️ [Memory Layer Warning]: Failed to sync interaction: {mem_err}")

    def get_relevant_history(self, user_id: str, query: str) -> dict[str, Any]:
        """
        Retrieve past interactions relevant to the current query.

        Args:
            user_id (str): Unique identifier for the customer.
            query (str): The customer's current query.

        Returns:
            List[Dict]: A list of relevant past interactions.
        """
      
        return self.memory.search(
            query=query,  
            filters={"user_id": user_id}, 
            limit=AGENT_RETRIEVAL_LIMIT  
        )

    def handle_customer_query(self, user_id: str, query: str) -> str:
        """
        Process a customer's query and provide a response, taking into account past interactions.

        :param user_id:
        :param query:
        :return:
        """

        # Retrieve relevant past memory facts
        relevant_history = self.get_relevant_history(user_id, query)

        # Normalize into a single iterable regardless of mem0's response shape
        if isinstance(relevant_history, dict):
            memories_list = relevant_history.get("results", [])
        elif isinstance(relevant_history, list):
            memories_list = relevant_history
        else:
            memories_list = []

        facts = [
            h['memory'] if isinstance(h, dict) and 'memory' in h else h
            for h in memories_list
            if isinstance(h, str) or (isinstance(h, dict) and 'memory' in h)
        ]

        # 3. Format the background profile for the Agent
        if facts:
            context_string = "\n".join(f"- {fact}" for fact in facts)
            context_header = f"Known user background profiles and preferences:\n{context_string}"
        else:
            context_header = "No prior user preferences or background profiles recorded."

        # 4. Structured input separates memory profile from the core question
        structured_input = f"""
        [USER METADATA PROFILE]
        {context_header}
    
        [CURRENT USER QUESTION]
        {query}
        """.strip()

        #logger.info('DEBUG: Executing agent invocation with structured input.')

        response = self.agent_executor.invoke({"input": structured_input})

        self.store_customer_interaction(
            user_id=user_id,
            message=query,
            response=response["output"],
            metadata={"type": "support_query"}
        )

        return response['output']

    def check_user_input(self, user_input: str) -> str:
        """
        Checks user input and returns an action.
        :param user_input:
        :return:
        """
        input_str = user_input.strip()

        # Define the logic for exiting the loop' [if the user types in exit]
        if input_str.lower() in AGENT_EXIT_CMDS:
            print(f"\n{I_SURPRISED} Agent: Goodbye! Feel free to return if you have more questions.\n")
            print(f"# --- Session End: {I_CLOCK} {show_datetime()} --- #")
            return 'break'

        # ℹ️ Note: If user just enters blank, skip Llama and ask for another query.
        elif len(input_str) == 0:
            print(f"{I_THUMBS_DOWN} Hey, you didn\'t say anything. Please ask a question.")
            return 'continue'

        else:
            return 'process'
