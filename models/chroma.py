from __future__ import annotations
# models/chroma.py

# Python Libraries
import logging
import os
import random
import time

import chromadb

# Vector Libraries
from langchain_chroma import Chroma
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_experimental.text_splitter import SemanticChunker

from src.constants import (
    CHROMA_SERVER_NO_TELEMETRY,
    CHROMA_VECTOR_RESULT_CNT,
    I_DOCUMENT,
    I_GEAR,
    I_INFO,
    I_PEN,
    I_PLUS,
    I_QUES,
    I_WARNING,
    SIMILARITY_SEARCH_QUERY,
)
from src.model_config import config, ModelConfig


class ChromaModel:
    """
    Manages the persistent vector store lifecycle using ChromaDB and handles both similarity-based and
    metadata-structured Self-Query retrieval mechanisms.

    see: https://docs.langchain.com/oss/python/langchain/rag?_gl=1*1jjuma6*_gcl_au*MTg3MjczODAzNS4xNzgxMjc3MDEx*_ga*MzU4MDUyMjIzLjE3ODEyNzcwMTE.*_ga_47WX3HKKY2*czE3ODE3NDE0MjkkbzMkZzEkdDE3ODE3NDE5NjkkajUyJGwwJGgw
    """

    def __init__(self, dataset: dict):

        # Force telemetry down
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = CHROMA_SERVER_NO_TELEMETRY
        logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

        # Ensures everything stays tightly isolated inside your db directory
        self.chromadb_client = chromadb.PersistentClient(path=os.path.abspath(config.CHROMA_VECTORS_DIR))

        #self.batch_size = 0 # remove
        self.collection_name = ''
        self.embedding_model = None 
        self.force_rebuild = False
        self.llm = None
        self.metadata_info = {} # remove
        self.title = ''

        # Set incoming pipeline dictionary variables
        self._set_attrs(dataset)

        # Build downstream storage partitions
        self.semantic_text_splitter = self._get_semantic_text_splitter()
        self.semantic_storage = self._get_semantic_storage()
        self.vector_storage = self._get_vector_storage()
        self.retriever = self.get_retriever()
       

    def _set_attrs(self, dataset: dict) -> None:
        """
        Safely maps dataset keys to class attributes, avoiding method overwrites.

        :param dataset:
        :return:
        """

        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.llm is None and 'openai_model' in dataset:
            openai_model = dataset['openai_model']
            self.llm = openai_model.llm

    @staticmethod
    def queries() -> list:
        return config.RETRIEVER_QUERIES

    def get_retriever(self) -> VectorStoreRetriever:
        """Initializes vector retriever using the unified client runtime pool."""

        # The underlying 'client' object safely handles the persistent directory paths now.
        return self.vector_storage.as_retriever(
            search_type="similarity",
            search_kwargs={"k": CHROMA_VECTOR_RESULT_CNT}
        )

    def _get_semantic_storage(self) -> Chroma:
        """
        Instantiates the isolated semantic database research partition.
        Purpose: This holds the real text from your PDF (nutritional-disorders.pdf), chopped up into clean sentences or paragraphs using your SemanticChunker.
        The Logic: When your AI agent needs to answer a user's medical question, it must pull facts from this collection to use as context.
        """
        print('DEBUG: Getting semantic storage with collection_name: semantic_chunks.')
        return Chroma(
            client=self.chromadb_client,
            embedding_function=self.embedding_model,
            collection_name="semantic_chunks"
        )

    def _get_vector_storage(self) -> Chroma:
        """
        Instantiates the generalized primary target layout collection.
        Purpose: This holds the synthetic data (the hypothetical questions your LLM generated from the text and tables).
 	    The Logic: This acts as a "Retriever Booster." Instead of matching a user's question directly to a dense block of textbook prose, the database matches the user's question to a hypothetical question that an LLM thought a human might ask. Matching Question-to-Question is mathematically much cleaner for embedding models than matching Question-to-Textbook-Paragraph.
        """
        print(f'DEBUG: Getting vector storage with collection_name: {self.collection_name}.')
        return Chroma(
            client=self.chromadb_client,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            #persist_directory=format_dir(self.collection_name)
        )

    def _get_semantic_text_splitter(self) -> SemanticChunker:

        return SemanticChunker(
            self.embedding_model,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=config.SEMANTIC_THRESH_LIMIT
        )

    def _get_structured_retriever(self) -> SelfQueryRetriever:
        print('DEBUG: Getting structured retreiever with vectorstore: self.semantic_storage.')
        return SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.semantic_storage,
            document_contents=config.RETRIEVER_DOCUMENT_CONTENT,
            metadata_field_info=config.RETRIEVER_DOCUMENT_METADATA_FIELDS,
            structured_query_translator=ChromaTranslator(),
            verbose=True,
            use_original_query=ModelConfig.is_premium()
        )

    def _get_structured_hyp_retriever(self) -> SelfQueryRetriever:
        print('DEBUG: Getting structured hyper retreiever with vectorstore: self.vector_storage.')
        return SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vector_storage,
            document_contents=config.RETRIEVER_HYPER_DOCUMENT_CONTENT,
            metadata_field_info=config.RETRIEVER_HYPER_DOCUMENT_METADATA_FIELDS,
            structured_query_translator=ChromaTranslator(),
            verbose=True,
            use_original_query=ModelConfig.is_premium()
        )

    def get_semantic_chunks(self, filepath: str) -> list:
        semantic_chunks = []
        pdf_loader = PyPDFLoader(filepath)
        chunks = pdf_loader.load_and_split(self.semantic_text_splitter)
        semantic_chunks.extend(chunks)

        return semantic_chunks

    def get_semantic_count(self) -> int:
        """Returns the number of documents in the semantic storage collection."""
        try:
            return self.semantic_storage._collection.count()
        except Exception:
            return 0

    def add_semantic_documents(self, semantic_chunks: list) -> None:
        batch_size = config.DOCUMENT_CHUNK_BATCH_SIZE 
        semantic_chunks_cnt = len(semantic_chunks)

        print(f'\n# --- {I_PLUS} Adding {semantic_chunks_cnt} semantic documents with a batch size of {batch_size} {I_PLUS}--- #')

        for i in range(0, semantic_chunks_cnt, batch_size):
            self.semantic_storage.add_documents(semantic_chunks[i: i + batch_size])

    def add_vector_documents(self, documents: list) -> None:
        batch_size = config.DOCUMENT_CHUNK_BATCH_SIZE
        document_cnt = len(documents)

        print(f'\n# --- {I_PLUS} Adding {document_cnt} vector documents with a batch size of {batch_size}. {I_PLUS} --- #')

        for i in range(0, document_cnt, batch_size):
            self.vector_storage.add_documents(documents[i : i + batch_size])

    def get_documents(self) -> list:
        return self.semantic_storage.similarity_search(
            query=SIMILARITY_SEARCH_QUERY,
            k=CHROMA_VECTOR_RESULT_CNT
        )

    def get_document_count(self) -> int:
        """Returns the number of documents currently in this collection."""
        try:
            return self.vector_storage._collection.count()
        except Exception:
            return 0

    def export(self) -> dict:
        """Exports attributes to be injected into child or dependent pipeline classes."""

        return {
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model,
            'force_rebuild': self.force_rebuild,
            'llm': self.llm,
        }

    def query_questions(self, is_hyp: bool=False, pluck: bool=False) -> None:
        """
        Query questions using either structured hypothetical retriever or structured retriever.

        :param is_hyp:
        :param pluck:
        :return:
        """

        retriever = self._get_structured_hyp_retriever() if is_hyp else self._get_structured_retriever()
        print(f'\n# --- {I_GEAR} Hypothetical Retriever {I_GEAR} --- #' if is_hyp else f'\n# --- {I_GEAR} Retriever {I_GEAR} --- #')

        queries_to_run = [random.choice(self.queries())] if pluck else self.queries()
        print(f'Number of queries to run: {len(queries_to_run)}')

        results_count = []
        for i, question in enumerate(queries_to_run):
            try:
                ques_semantic_chunks_retrieved = retriever.invoke(question)
                retrieved_count = len(ques_semantic_chunks_retrieved)

                fallback_str = ""

                # Display empty retrieval
                if retrieved_count == 0:
                    print(f'{I_WARNING}  Nothing was retrieved! Swapping to vector similarity as a fallback... {I_WARNING}')

                    # Fallback with the regular retriever
                    ques_semantic_chunks_retrieved = self.retriever.invoke(question)
                    retrieved_count = len(ques_semantic_chunks_retrieved)

                    print(f"\n----- {I_QUES} Question #{i+1} {I_QUES} -----")
                    print(question)
                    print(f"\n{I_INFO}  Number of Semantic Chunks Retrieved: {retrieved_count}")

                    if retrieved_count == 0:
                        print(f'{I_WARNING}  Again, nothing was retrieved across fallback storage layer indexes! {I_WARNING}')
                    else:
                        fallback_str = "Fallback"
                        #print(f"{I_DOCUMENT} Fallback Retrieved Documents:\n{ques_semantic_chunks_retrieved}")
                    
                #results_count.append(retrieved_count)
                else:
                    print(f"\n----- {I_QUES} Question #{i+1} {I_QUES} -----")
                    print(question)
                    print(f"\n{I_INFO}  Number of Semantic Chunks Retrieved: {retrieved_count}")

                        
                #else:
                print(f"{I_DOCUMENT} {fallback_str} Retrieved Documents:\n{ques_semantic_chunks_retrieved}")
                results_count.append(retrieved_count)

            except Exception as e:
                print(f"⚠️ Skipping query due to parser error: {question}.")
                print(f"\tReason: {e}")
                results_count.append(0)

            print(f"+---- Question #{i+1} ----+\n")
            time.sleep(config.SLEEP_TIME_SEC)

        total = len(queries_to_run)
        successful = sum(1 for c in results_count if c > 0)
        print(f"{I_PEN} Retriever Quality: {successful}/{total} queries returned results ({successful/total*100:.0f}%)")
