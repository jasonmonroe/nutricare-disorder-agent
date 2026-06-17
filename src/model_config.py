# src/model_config.py

# Python Libraries
import sys

from src.constants import (
    I_CROSSMARK, 
    I_SKULL, 
    LLAMA_MODEL, 
    OPENAI_EMBEDDING_MODEL, 
    OPENAI_MODEL
    )
import src.constants_free as c_free 
import src.constants_premium as c_prem


class ModelConfig:
    """
    This project was originally written with specific models that worked fine.  Due to de-commissioning from the vendor we
    are now forced to use inferior models that requires a different configuration setup as well as distinctive prompts to
    obtain the same performance.

    We will define each constant based on the models and then use a configuration script to know which constant to utilize.
    """

    @classmethod
    def is_premium(cls) -> bool:
        # use cls to represent 'class itself'
        return LLAMA_MODEL == c_prem.PREMIUM_LLAMA_MODEL and OPENAI_EMBEDDING_MODEL == c_prem.PREMIUM_OPENAI_EMBEDDING_MODEL and OPENAI_MODEL == c_prem.PREMIUM_OPENAI_MODEL

    @staticmethod
    def check_models(chroma_db, openai_model, llama) -> None:
        if chroma_db is None or openai_model is None or llama is None:
            print(f"{I_CROSSMARK} Core dependencies didn't load properly. Exiting system!!! {I_CROSSMARK}")
            print(f'{I_SKULL}')
            sys.exit(0)

    @staticmethod
    def check_chroma_db(chroma_db) -> None:
        semantic_count = chroma_db.get_semantic_count()
        vector_count = chroma_db.get_document_count()

        if semantic_count > 0 or vector_count > 0:
            print(f'Semantic: {semantic_count}, Vectors: {vector_count}')

        if semantic_count == 0 and vector_count == 0:
            print(f"{I_CROSSMARK} Out of sync! Both vector partitions are completely empty. Please run with --data first! {I_CROSSMARK}")
            raise RuntimeError("Vector database contains zero records across all internal collections.")

        if vector_count == 0:
            print(f"{I_CROSSMARK} No documents found in the vector storage. Please run with --data first! {I_CROSSMARK}")
            raise RuntimeError("Vector database is completely empty!")

# Set the global config variable
config = c_prem if ModelConfig.is_premium() else c_free
