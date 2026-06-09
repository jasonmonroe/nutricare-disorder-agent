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
#from langchain_core.output_parsers import StrOutputParser, JsonOutputParser  # String output parser
from langchain_core.prompts import ChatPromptTemplate as CoreChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# Local Libraries
from src.config import I_THUMBS_DOWN, INACTIVE_SESSION_DUR, MEM0_API_KEY, RETRIEVAL_LIMIT
from src.utils import get_time, start_timer


class NutritionBot:
    def __init__(self, llm_chatbot, tools: list):
        """
         Initialize the NutritionBot class, setting up memory, the LLM client, tools, and the agent executor.

        :param llm_chatbot:
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
            print(f'DEBUG: diff_in_secs={diff_in_secs}')
            print(f'\n{I_THUMBS_DOWN} Session has expired.  Exiting chat.')
            return True

        return False

    def get_session_duration(self, session_ends_at) -> str:
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

        self.memory.add(
            conversation,
            filters={"user_id": user_id},
            output_format="v1.1",
            metadata=metadata
        )

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
            limit=RETRIEVAL_LIMIT  
        )

    def handle_customer_query(self, user_id: str, query: str) -> str:
        """
        Process a customer's query and provide a response, taking into account past interactions.

        Args:
            user_id (str): Unique identifier for the customer.
            query (str): Customer's query.

        Returns:
            str: Chatbot's response.
        """
        logger = logging.getLogger(__name__)

        # Retrieve relevant past interactions for context
        relevant_history = self.get_relevant_history(user_id, query)
        logger.info(f'relevant_history={relevant_history}')
        

        # Build a context string from the relevant history
        context = "Previous relevant interactions:\n"

        """
        for history in relevant_history:
            logger.info(f'history={history}')
            context += f"Customer: {history['memory']}\n"  # Customer's past messages
            context += f"Support: {history['memory']}\n"  # Chatbot's past responses
            context += "---\n"

        
        memories = relevant_history.get('results', [])
        logger.info(f'memories={memories}')

        if memories:
            context += "\nPast messages and responses:\n"
            for item in memories:
                memory_text = item.get('memory', '')
                if memory_text:
                    context += f'- {memory_text}'
        else:
            context += "\nNo relevant past profile history found.\n"
        """

        
        if isinstance(relevant_history, dict):
            # Target the inner list under the 'results' key (defaults to an empty list if missing)
            memories_list = relevant_history.get("results", [])
            for history in memories_list:
                logger.info(f'history object found in dict: {history}')
                if isinstance(history, dict) and 'memory' in history:
                    context += f"- Context Fact: {history['memory']}\n"
                elif isinstance(history, str):
                    context += f"- Context Fact: {history}\n"
                    
        elif isinstance(relevant_history, list):
            # Fallback in case a different version/mock payload returns a flat list directly
            for history in relevant_history:
                logger.info(f'history object found in list: {history}')
                if isinstance(history, dict) and 'memory' in history:
                    context += f"- Context Fact: {history['memory']}\n"
                elif isinstance(history, str):
                    context += f"- Context Fact: {history}\n"
       

        # Print context for debugging purposes
        #logger.info("Context: ", context)
        logger.info(f"Context Compiled Successfully:\n {context}")

        # Prepare a prompt combining past context and the current query
        prompt = f"""
        Context:
        {context}

        Current customer query: {query}

        Provide a helpful response that takes into account any relevant past interactions.
        """.strip()

        logger.info(f'line 219 DEBUG: prompt:{prompt}')
        
        # Generate a response using the agent
        response = self.agent_executor.invoke({"input": prompt})

        # Store the current interaction for future reference
        self.store_customer_interaction(
            user_id=user_id,
            message=query,
            response=response["output"],
            metadata={"type": "support_query"}
        )

        # Return the chatbot's response
        return response['output']
