# tools/agentic_rag.py

# +---------------------+
# |     AGENTIC RAG     |
# +---------------------+

# Vendor Libraries
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph.state import CompiledStateGraph

# Local Libraries
from models.agentic_rag_tool import AgenticRagTool
from src.config import AI_ROLE


def make_agentic_rag_tool(llm: ChatOpenAI, retriever: VectorStoreRetriever, workflow_app) -> StructuredTool:
    """Create an agentic RAG tool with llm and retriever bound to the closure."""

    @tool
    def agentic_rag(query: str):
        """
        Runs the RAG-based agent with conversation history for context-aware responses.

        Args:
            query (str): The current user query.

        Returns:
            Dict[str, Any]: The updated state with the generated response and conversation history.
        """
        inputs = {
            "query": query,
            "expanded_query": "",
            "context": [],
            "response": "",
            "precision_score": 0.0,
            "groundedness_score": 0.0,
            "groundedness_loop_count": 0,
            "precision_loop_count": 0,
            "feedback": "",
            "query_feedback": "",
            "loop_max_iter": 4,
            "AI_ROLE": AI_ROLE,
        }
        
        if isinstance(workflow_app, CompiledStateGraph):
            #print(f'line 48: DEBUG: Using param: workflow_app type={type(workflow_app)}')
            return workflow_app.invoke(inputs)

        # If workflow_app is not found or None, build the agent
        agentic_rag_tool = AgenticRagTool(llm, retriever)
        workflow_app = agentic_rag_tool.compile()
        #print(f'line 54: DEBUG: workflow_app type={type(workflow_app)}')
        
        return workflow_app.invoke(inputs)

    # Returns here if --start is in args
    return agentic_rag
