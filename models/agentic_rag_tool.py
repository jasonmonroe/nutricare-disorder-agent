# models/agentic_rag_tool.py

# Python Libraries
import json
from typing import Dict, List
from pydantic import BaseModel
from IPython.display import Image, display

# Vendor Libraries
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate as CoreChatPromptTemplate
from langgraph.graph import StateGraph, END, START  # State graph for managing states in LangChain

# Local
from models.agentic_state import AgentState
from src.config import AI_ROLE, EVAL_THRESHOLD


class AgenticRagTool:
    def __init__(self, llm, retriever):

        self.llm = llm
        self.retriever = retriever

    def compile(self):
        return self._create().compile()

    #
    def _create(self) -> StateGraph:
        """
        Used for LineGraph (library of Agentic RAG), a workflow is modeled as a StateGraph, which is simply a state
        machine (like a complex flowchart).
        :return: StateGraph
        """

        """Creates the updated workflow for the AI nutrition agent."""
        workflow = StateGraph(AgentState)

        # Add processing nodes
        workflow.add_node("expand_query", self.expand_query)                     # Step 1: Expand user query.
        workflow.add_node("retrieve_context", self.retrieve_context)             # Step 2: Retrieve relevant documents.
        workflow.add_node("craft_response", self.craft_response)                 # Step 3: Generate a response based on retrieved data.
        workflow.add_node("score_groundedness", self.score_groundedness)         # Step 4: Evaluate response grounding.
        workflow.add_node("refine_response", self.refine_response)               # Step 5: Improve response if it's weakly grounded.
        workflow.add_node("check_precision", self.check_precision)               # Step 6: Evaluate response precision.
        workflow.add_node("refine_query", self.refine_query)                     # Step 7: Improve query if response lacks precision.
        workflow.add_node("max_iterations_reached", self.max_iterations_reached) # Step 8: Handle max iterations.

        # Define the entry point where to start
        workflow.set_entry_point("expand_query")

        # Main flow edges
        workflow.add_edge("expand_query", "retrieve_context")
        workflow.add_edge("retrieve_context", "craft_response")
        workflow.add_edge("craft_response", "score_groundedness")

        # Conditional edges based on groundedness check
        workflow.add_conditional_edges(
            "score_groundedness",
            self.should_continue_groundedness,  # Use the conditional function
            {
                "check_precision": "check_precision",              # If well-grounded, proceed to precision check.
                "refine_response": "refine_response",                # If not, refine the response.
                "max_iterations_reached": "max_iterations_reached" # If max loops reached, exit.
            }
        )

        workflow.add_edge("refine_response", "craft_response")  # Refined responses are reprocessed.

        # Conditional edges based on precision check
        workflow.add_conditional_edges(
            "check_precision",
            self.should_continue_precision,                        # Use the conditional function
            {
                "pass": END,                                       # If precise, complete the workflow.
                "refine_query": "refine_query",                      # If imprecise, refine the query.
                "max_iterations_reached": "max_iterations_reached" # If max loops reached, exit.
            }
        )

        workflow.add_edge("refine_query", "expand_query") # Refined queries go through expansion again.
        workflow.add_edge("max_iterations_reached", END)

        return workflow
            

    # --- MAX ITERATIONS REACHED
    def max_iterations_reached(self, state: AgentState) -> AgentState:
        """
        Handles the case where max iterations are reached.

        Args:
        :param state: state of agent
        :return: returns state response
        """
         
        state['response'] = "We need more context to provide an accurate answer."
        return state


    def expand_query(self, state: AgentState) -> AgentState:
        """
        Expands the user query to improve retrieval of nutrition-disorder-related information using few-shot prompting.

        Args:
            state (Dict): The current state of the workflow, containing the user query.

        Returns:
            Dict: The updated state with the expanded query.
        """

        print("\n-------- expand_query ---------")

        original_query = state['query']
        query_feedback = state.get('query_feedback') # Gets feedback if present

        # --- Start with the ROBUST V1 Prompt ---
        system_message = f"""
        You are an expert AI {AI_ROLE} specializing in nutritional disorders and academic literature search.

        Your task is to rewrite the user's query into a single detailed, precise, and technical search query optimized for retrieving relevant academic research papers on nutrition disorders.

        - Incorporate key domain-specific terms, synonyms, and related clinical terminology.
        - Expand abbreviations and clarify ambiguous terms with terminology common in scientific literature.
        - Keep the core clinical intent and meaning intact.

        - **CRITICAL EXCEPTION (V1 INTEGRATION):** If the user's query is **conversational**, **procedural**, or **meta-data related** (e.g., "What can I ask?", "Who are you?"), **DO NOT** expand it. **Return the original user query exactly as provided.**

        - Format the output as a concise query string suitable for academic database search engines.
        - Your output MUST be only the rewritten query string and nothing else.
        """

        if query_feedback:
            print("--- Using feedback to refine query ---")

            system_message += f"""

            You have already generated a query that was not precise enough. Use the following SUGGESTIONS to create a NEW, improved query.

            SUGGESTIONS:
            {query_feedback}
            """

        # Create the final prompt template
        expand_prompt = CoreChatPromptTemplate.from_messages([
            ("system", system_message.strip()),
            ("user", "Original User Query: {query}")
        ])

        chain = expand_prompt | self.llm | StrOutputParser()

        # Invoke the chain
        state['expanded_query'] = chain.invoke({
            "query": original_query,
            # Note: Feedback is injected via the system_message,
            "ROLE": state['AI_ROLE'],
        })

        # Clear the feedback for the next node
        state['query_feedback'] = ""

        return state


    # --- RETRIEVE CONTEXT
    def retrieve_context(self, state: AgentState) -> AgentState:
        """
        Retrieves context from the vector store using the expanded or original query.

        Args:
            state (Dict): The current state of the workflow, containing the query and expanded query.

        Returns:
            Dict: The updated state with the retrieved context.
        """

        query = state['expanded_query']

        print("\n--- retrieve_context ---")
        print("Query used for retrieval:", query)  # Debugging: Print the query

        # Retrieve documents from the vector store
        retrieved_docs = self.retriever.invoke(query)

        print("Retrieved documents:", retrieved_docs)  # Debugging: Print the raw docs object

        # Extract both page_content and metadata from each document
        state['context'] = [
            {
                "content": doc.page_content,  # The actual content of the document
                "metadata": doc.metadata  # The metadata (e.g., source, page number, etc.)
            }
            for doc in retrieved_docs
        ]

        print("Extracted context with metadata:", state['context'])  # Debugging: Print the extracted context

        return state


    # --- CRAFT RESPONSE
    def craft_response(self, state: Dict) -> Dict:
        """
        Generates a response using the retrieved context, focusing on nutrition disorders.

        Args:
            state (Dict): The current state of the workflow, containing the query and retrieved context.

        Returns:
            Dict: The updated state with the generated response.
        """
        print("\n--- craft_response ---")

        system_message = """
        You are an expert AI {ROLE}, specializing in **Nutritional Disorders**. Your sole task is to analyze the provided CONTEXT and synthesize a direct, comprehensive answer to the user's QUERY.

        **STRICT GENERATION RULES:**
        1.  **Groundedness:** Generate the response using **ONLY** the information found in the retrieved CONTEXT. Do not use outside knowledge.
        2.  **Format:** Output the answer as a structured, numbered list of concise, clinically relevant statements.
        3.  **Incorporation:** If the 'FEEDBACK' suggests improvements, use it to refine the response based on the CONTEXT. If there is no FEEDBACK, ignore it.
        4.  **Clinical Detail:** Include **key numeric thresholds**, **dosage recommendations**, and **specific diagnostic criteria** exactly as they appear in the CONTEXT.
        5.  **Citations:** Append the source document/page number to each statement if available in the CONTEXT.

        **INCOMPLETENESS:**
        If the retrieved CONTEXT is insufficient to answer the query, your entire response must be: **"The retrieved context is insufficient to provide a complete and grounded answer. Further clinical follow-up is recommended."**

        **DO NOT** include any commentary, greetings, or introductory/concluding remarks outside of the numbered list.
        """

        response_prompt = CoreChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "Query: {query}\nContext: {context}\n\nfeedback: {feedback}")
        ])

        chain = response_prompt | self.llm
        response = chain.invoke({
            "query": state['query'],
            "context": "\n".join([doc["content"] for doc in state['context']]),
            "feedback": state["feedback"], # add feedback to the prompt
            "ROLE": state["ROLE"],
        })

        state['response'] = response

        print("intermediate response: ", response)

        return state


    # --- SCORE GROUNDEDNESS
    def score_groundedness(self, state: Dict) -> Dict:
        """
        Checks whether the response is grounded in the retrieved context.

        Args:
            state (Dict): The current state of the workflow, containing the response and context.

        Returns:
            Dict: The updated state with the groundedness score.
        """

        print("\n--- check_groundedness ---")

        system_message = """You are a meticulous AI {ROLE} Quality Analyst and fact-checker. Your sole task is to evaluate how well a given response is supported by a provided context.
        Calculate a score from 0.0 to 1.0 that represents the fraction of claims in the response that are directly and verifiably supported by the context.
        - A score of 1.0 means every claim in the response is fully supported by the context.
        - A score of 0.0 means no claims in the response are supported by the context.

        **Your output MUST be the numerical score as a single float. Do not output any other text, explanation, or markdown.**
        """

        groundedness_prompt = CoreChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "Context: {context}\nResponse: {response}\n\nGroundedness score:")
        ])

        chain = groundedness_prompt | self.llm | StrOutputParser()
        groundedness_score = float(chain.invoke({
            "context": "\n".join([doc["content"] for doc in state['context']]),
            "response": state['response'],
            "ROLE": state["ROLE"],
        }))


        state['groundedness_loop_count'] += 1

        print("groundedness_score: ", groundedness_score)
        print("######## Groundedness Incremented ##########")

        state['groundedness_score'] = groundedness_score

        return state


    # --- CHECK PRECISION
    def check_precision(self, state: Dict) -> Dict:
        """
        Checks whether the response precisely addresses the user’s query.

        Args:
            state (Dict): The current state of the workflow, containing the query and response.

        Returns:
            Dict: The updated state with the precision score.
        """

        print("\n--- check_precision ---")

        system_message = """
        As an AI {ROLE} evaluate whether the response precisely addresses the user's query.
        Evaluate, assign and return the precision score for the response.  Your evaluation is based solely on the relationship between the response and the query. Do not consider anything else.

        The score is from 0.0 (least) to 1.0 (best).
        - A score of 1.0 means the response is precise, on-topic and accurately addressed by the query.
        - A score of 0.0 means the response is ambiguous, off-topic, or inaccurate and does not accurately address the query.

        **Only output the numerical score as a float and nothing else!**
        """

        precision_prompt = CoreChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "Query: {query}\nResponse: {response}\n\nPrecision score:")
        ])

        chain = precision_prompt | self.llm | StrOutputParser()
        precision_score = float(chain.invoke({
            "query": state['query'],
            "response": state['response'],
            "ROLE": state["ROLE"],
        }))

        state['precision_score'] = precision_score
        state['precision_loop_count'] += 1

        print("precision_score:", precision_score)
        print("# --- Precision Incremented --- #")

        return state


    # --- REFINE RESPONSE
    def refine_response(self, state: Dict) -> Dict:
        """
        Suggests improvements for the generated response.

        Args:
            state (Dict): The current state of the workflow, containing the query and response.

        Returns:
            Dict: The updated state with response refinement suggestions.
        """

        print("\n--- refine_response ---")

        system_message = """
        You are an AI {ROLE} Quality Analyst and Critic. Your sole task is to provide constructive feedback on a given response based on the user's original query.
        Your feedback should identify potential gaps, ambiguities, or missing details and suggest specific improvements to enhance the response's accuracy and completeness.

        - Use bullet points to structure your suggestions.
        - Do NOT rewrite the full response. Only provide a list of suggestions for improvement.
        - Your output must be only the bulleted list of suggestions. Do not include a preamble like "Here are my suggestions:".
        """

        refine_response_prompt = CoreChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "Query: {query}\nResponse: {response}\n\n"
                    "What improvements can be made to enhance accuracy and completeness?")
        ])

        chain = refine_response_prompt | self.llm | StrOutputParser()

        # Store response suggestions in a structured format
        feedback = f"Previous Response: {state['response']}\nSuggestions: {chain.invoke({'query': state['query'], 'response': state['response'], 'ROLE': state['ROLE']})}"

        print("feedback: ", feedback)
        print(f"State: {state}")

        state['feedback'] = feedback

        return state


    # --- REFINE QUERY
    def refine_query(self, state: Dict) -> Dict:
        """
        Suggests improvements for the expanded query, returning them in a structured JSON format.

        Args:
            state (Dict): The current state of the workflow, containing the query and expanded query.

        Returns:
            Dict: The updated state with JSON-formatted query refinement suggestions.
        """

        print("\n--- refine_query ---")

        # Define a Pydantic model that matches the desired JSON structure.
        # This is the correct way to provide a schema to JsonOutputParser.
        class QuerySuggestions(BaseModel):
            missing_keywords: List[str]
            scope_refinements: List[str]
            term_clarifications: List[str]

        # This prompt forces the JSON structure and ensures high-quality clinical input
        system_message = f"""
        You are an AI Search Query Analyst specializing in clinical nutrition literature.
        Your sole task is to provide constructive feedback on the provided expanded query to enhance its search precision for academic databases.

        - Do NOT rewrite the query.
        - Analyze the Expanded Query against the Original Query and suggest improvements only in the required JSON format.
        - If a category has no suggestions, return an empty list for that key.

        - Your output MUST be a JSON object that strictly adheres to the format defined by the tool.
        """
        system_message = system_message.strip()

        # Use the LangChain JsonOutputParser for reliable structured output
        json_parser = JsonOutputParser(pydantic_object=QuerySuggestions)

        refine_query_prompt = CoreChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "Original Query: {query}\nExpanded Query to Critique: {expanded_query}\n\nProvide your JSON suggestions:")
        ])

        chain = refine_query_prompt | self.llm | json_parser

        # Invoke the chain to get structured suggestions
        suggestions = chain.invoke({
            "query": state['query'],
            "expanded_query": state['expanded_query'],
            "ROLE": state["ROLE"],
        })

        # Store the JSON object as a string in the state for the next node to consume
        suggestions_str = json.dumps(suggestions, indent=2)

        state['query_feedback'] = suggestions_str

        print(f"Query Feedback Generated (JSON):\n{suggestions_str}")

        return state


    # --- HAS MAX ITERATIONS REACHED?
    # Checks if the maximum number of iterations has been reached
    # Note: This method must be before should_* methods.
    def has_max_iterations_reached(self, state: Dict, var: str) -> bool:
        return state[var] >= state["loop_max_iter"]


    # --- CHECK GROUNDEDNESS
    def should_continue_groundedness(self, state) -> str:
        """
        Decides if groundedness is enough or needs improvement.

        Args:
        :param state:
        :return: string of next node to invoke
        """
        """Decides if groundedness is enough or needs improvement."""

        print("--- should_continue_groundedness ---")
        print("groundedness loop count: ", state['groundedness_loop_count'])

        if state["groundedness_score"] >= EVAL_THRESHOLD:  # Threshold for groundedness
            print("Moving to precision")

            return "check_precision"

        else:
            if self.has_max_iterations_reached(state, "groundedness_loop_count"):
                return "max_iterations_reached"
            else:
                print("--- Groundedness Score Threshold Not met. Refining Response -----")

                return "refine_response"


    # --- CHECK PRECISION
    def should_continue_precision(self, state: Dict) -> str:
        """
        Decides if precision is enough or needs improvement.

        Args:
        :param state:
        :return string of next node to invoke
        """
        """Decides if precision is enough or needs improvement."""

        print("--- should_continue_precision ---")
        print("precision loop count: ", state['precision_loop_count'])

        if state["precision_score"] >= EVAL_THRESHOLD:  # Threshold for precision
            return "pass"  # Complete the workflow

        else:
            if self.has_max_iterations_reached(state, "precision_loop_count"):  # Maximum allowed loops
                return "max_iterations_reached"
            else:
                print("--- Precision Score Threshold not met. Refining Query ---")

                return "refine_query"  # Refine the query


    # --- MAX ITERATIONS REACHED
    def max_iterations_reached(self, state: AgentState) -> AgentState:
        """
        Handles the case where max iterations are reached.

        Args:
        :param state:
        :return:
        """

        state['response'] = "We need more context to provide an accurate answer."
        return state


    def display_workflow(self, app) -> None:
        display(Image(app.get_graph().draw_mermaid_png()))
