# pipelines/agent.py
import nest_asyncio
from models.chroma import ChromaModel
from models.llama import LlamaModel
from models.openai import OpenAIModel


def build(show_logs: bool=False):

    # Initialize streamlit persistent state
    print(f'DEBUG: show_logs:{show_logs}')

    openai_model = OpenAIModel()
    llm = openai_model.load_llm()
    llama = LlamaModel(llm, openai_model.embedding_model)

    # --- INITIALIZE CHROMA VECTOR STORAGE FOR RETRIEVING DOCUMENTS
    # Retrieve `nutritional` database created from Google Colab

    # Create vector storage for nutritional information
    chroma_db = ChromaModel({
        'llm': llm,
        'embedding_model': openai_model.embedding_model,
        'collection_name': 'nutritional'
    })

    # Stage 2 - Start Program
    # Apply the nested async loop to allow async code execution in the notebook
    nest_asyncio.apply()

    # --- Visualize Workflow --- #
    workflow_app = None






def start():
    pass
