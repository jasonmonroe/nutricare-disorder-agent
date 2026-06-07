# tools/agentic_rag.py

# +---------------------+
# |     AGENTIC RAG     |
# +---------------------+

# Vendor Libraries
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever

# Local Libraries
from models.agentic_rag_tool import AgenticRagTool
from src.config import AI_ROLE


def make_agentic_rag_tool(llm: ChatOpenAI, retriever: VectorStoreRetriever):
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

        agentic_rag_tool = AgenticRagTool(llm, retriever)
        workflow_app = agentic_rag_tool.compile()
        return workflow_app.invoke(inputs)

    return agentic_rag
