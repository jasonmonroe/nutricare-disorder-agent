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
    metadata-structured Self-Query retrieval mechanisms using a clean, flat architecture.

    see: https://docs.langchain.com/oss/python/langchain/rag?_gl=1
    """

    def __init__(self, dataset: dict):

        # Force telemetry down
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = CHROMA_SERVER_NO_TELEMETRY
        logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

        # Ensures everything stays tightly isolated inside your db directory
        self.chromadb_client = chromadb.PersistentClient(path=os.path.abspath(config.CHROMA_VECTORS_DIR))

        self.collection_name = ''
        self.embedding_model = None
        self.force_rebuild = False
        self.llm = None
        self.title = ''

        # Set incoming pipeline dictionary variables
        self._set_attrs(dataset)

        # Build downstream storage partitions
        self.semantic_text_splitter = self._get_semantic_text_splitter()
        self.semantic_storage = self._get_semantic_storage()
        self.vector_storage = self._get_vector_storage()
        self.retriever = self.get_retriever()


    def _set_attrs(self, dataset: dict) -> None:
        """Safely maps dataset keys to class attributes, avoiding method overwrites."""
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
        return self.vector_storage.as_retriever(
            search_type="similarity",
            search_kwargs={"k": CHROMA_VECTOR_RESULT_CNT}
        )

    def _get_semantic_storage(self) -> Chroma:
        """Instantiates the isolated semantic database research partition for textbook prose."""

        return Chroma(
            client=self.chromadb_client,
            embedding_function=self.embedding_model,
            collection_name="semantic_chunks"
        )

    def _get_vector_storage(self) -> Chroma:
        """Instantiates the primary flattened synthetic layout collection."""
        return Chroma(
            client=self.chromadb_client,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
        )

    def _get_semantic_text_splitter(self) -> SemanticChunker:
        return SemanticChunker(
            self.embedding_model,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=config.SEMANTIC_THRESH_LIMIT
        )

    def _get_structured_retriever(self) -> SelfQueryRetriever:

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
        """Load raw documents directly and invoke splitting via the semantic chunker instance."""
        pdf_loader = PyPDFLoader(filepath)
        raw_documents = pdf_loader.load()
        return self.semantic_text_splitter.split_documents(raw_documents)

    def get_semantic_count(self) -> int:
        """Returns the number of documents in the semantic storage collection."""
        try:
            return self.semantic_storage._collection.count()
        except Exception:
            return 0

    def add_semantic_documents(self, semantic_chunks: list) -> None:
        batch_size = config.DOCUMENT_CHUNK_BATCH_SIZE
        semantic_chunks_cnt = len(semantic_chunks)

        print(f'\n# --- {I_PLUS} Adding {semantic_chunks_cnt} semantic documents with a batch size of {batch_size} {I_PLUS} --- #')

        for i in range(0, semantic_chunks_cnt, batch_size):
            self.semantic_storage.add_documents(semantic_chunks[i: i + batch_size])

    def add_vector_documents(self, documents: list) -> None:
        batch_size = config.DOCUMENT_CHUNK_BATCH_SIZE
        document_cnt = len(documents)

        print(f'\n# --- {I_PLUS} Adding {document_cnt} vector documents with a batch size of {batch_size}. {I_PLUS} --- #')

        for i in range(0, document_cnt, batch_size):
            self.vector_storage.add_documents(documents[i : i + batch_size])

    def get_documents(self, query: str = SIMILARITY_SEARCH_QUERY) -> list:
        """
        🎯 SIMPLIFIED FLAT RETRIEVAL:
        Queries the flattened target collection directly. Because data chunks are bundled together
        at ingestion, a single hit returns questions and original context with zero cross-collection lookup logic.
        """
        return self.vector_storage.similarity_search(
            query=query,
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
        """Query questions using either structured hypothetical retriever or structured retriever."""
        retriever = self._get_structured_hyp_retriever() if is_hyp else self._get_structured_retriever()
        print(f'\n# --- {I_GEAR} Hypothetical Retriever {I_GEAR} --- #' if is_hyp else f'\n# --- {I_GEAR} Retriever {I_GEAR} --- #')

        fallback_retriever = (
            self.vector_storage.as_retriever(search_type="similarity", search_kwargs={"k": CHROMA_VECTOR_RESULT_CNT})
            if is_hyp else
            self.semantic_storage.as_retriever(search_type="similarity", search_kwargs={"k": CHROMA_VECTOR_RESULT_CNT})
        )

        queries_to_run = [random.choice(self.queries())] if pluck else self.queries()
        print(f'\tNumber of queries to run: {len(queries_to_run)}')

        results_count = []
        for i, question in enumerate(queries_to_run):
            try:
                ques_semantic_chunks_retrieved = retriever.invoke(question)
                retrieved_count = len(ques_semantic_chunks_retrieved)
                fallback_str = ""

                if retrieved_count == 0:
                    print(f'\t{I_WARNING}  Nothing was retrieved! Swapping to vector similarity as a fallback... {I_WARNING}')

                    # Fallback with the matching-collection retriever
                    ques_semantic_chunks_retrieved = fallback_retriever.invoke(question)
                    retrieved_count = len(ques_semantic_chunks_retrieved)

                    print(f"\n\t----- {I_QUES} Question #{i+1} {I_QUES} -----")
                    print("\t" + question)
                    print(f"\n\t{I_INFO}  Number of Semantic Chunks Retrieved: {retrieved_count}")

                    if retrieved_count == 0:
                        print(f'\t{I_WARNING}  Again, nothing was retrieved across fallback storage layer indexes! {I_WARNING}')
                    else:
                        fallback_str = "Fallback"
                else:
                    print(f"\n\t----- {I_QUES} Question #{i+1} {I_QUES} -----")
                    print("\t" + question)
                    print(f"\n\t{I_INFO}  Number of Semantic Chunks Retrieved: {retrieved_count}")

                print(f"\t{I_DOCUMENT} {fallback_str} Retrieved Documents:\n\t{ques_semantic_chunks_retrieved}")
                results_count.append(retrieved_count)

            except Exception as e:
                print(f"\t⚠️ Skipping query due to parser error: {question}.")
                print(f"\t\tReason: {e}")
                results_count.append(0)

            print(f"\t+---- Question #{i+1} ----+\n")
            time.sleep(config.SLEEP_TIME_SEC)

        total = len(queries_to_run)
        successful = sum(1 for c in results_count if c > 0)
        print(f"\t{I_PEN} Retriever Quality: {successful}/{total} queries returned results ({successful/total*100:.0f}%)")
