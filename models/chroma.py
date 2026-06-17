# models/chroma.py

# Python Libraries
import logging
import os
import random
import time

import chromadb

# Vector Libraries
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.query_constructors.chroma import ChromaTranslator

from src.constants import (
    CHROMA_SERVER_NO_TELEMETRY,
    CHROMA_VECTOR_RESULT_CNT,
    CHROMA_VECTORS_DIR,
    DOCUMENT_CHUNK_BATCH_SIZE,
    DOCUMENT_FILE,
    DOCUMENT_FILEPATH,
    I_DOCUMENT,
    I_GEAR,
    I_INFO,
    I_QUES,
    I_WARNING,
    RATE_LIMIT_TIME,
    SEMANTIC_THRESH_LIMIT,
    SIMILARITY_SEARCH_QUERY,
    SLEEP_TIME_SEC, 
)
from src.utils import format_dir


class ChromaModel:
    """
    Manages the persistent vector store lifecycle using ChromaDB and handles both similarity-based and
    metadata-structured Self-Query retrieval mechanisms.
    """

    def __init__(self, dataset: dict):

        # Force telemetry down
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = CHROMA_SERVER_NO_TELEMETRY
         
        logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

        # Ensures everything stays tightly isolated inside your db directory
        self.chromadb_client = chromadb.PersistentClient(path=os.path.abspath(CHROMA_VECTORS_DIR))

        #self.batch_size = 0 # remove
        self.collection_name = ''
        #self.document_content_description = '' # remove
        self.embedding_model = None 
        self.force_rebuild = False
        self.llm = None
        self.metadata_info = {} # remove
        self.title = ''

        # Set incoming pipeline dictionary variables
        self._set_attrs(dataset)

        # Build downstream storage partitions
        self.semantic_storage = self._get_semantic_storage()
        self.vector_storage = self._get_vector_storage()
        self.retriever = self.get_retriever()
        self.semantic_text_splitter = self._get_semantic_text_splitter()
        #self.structured_retriever = self._get_structured_retriever()
        #self.structured_hyp_retriever = self._get_structured_hyp_retriever()

    def _set_attrs(self, dataset: dict) -> None:
        """
        Safely maps dataset keys to class attributes, avoiding method overwrites.

        :param dataset:
        :return:
        """

        for key, value in dataset.items():
            if hasattr(self, key):
                #print(f'DEBUG: key={key}, value={value}')
                setattr(self, key, value)

        if self.llm is None and 'openai_model' in dataset:
            openai_model = dataset['openai_model']
            self.llm = openai_model.llm


    @staticmethod
    def queries() -> list:
        return [
            "What is the recommended dosage for treating scurvy?",
            "What is the definition of Vitamin C, based only on content from pages 1 through 14?",
            f"Describe the clinical signs of nutritional deficiency documented in {DOCUMENT_FILEPATH}.",
            "What type of nutritional support is needed for patients to increase lean body mass?",
            "On page 35, please explain the correlation between Vitamin B12 levels and tissue deficiency.",
            "What are the laboratory and clinical standards for diagnosing Vitamin D deficiency in adults?",
        ]

    def get_retriever(self) -> VectorStoreRetriever:
        """Initializes vector retriever using the unified client runtime pool."""

        # The underlying 'client' object safely handles the persistent directory paths now.
        return self.vector_storage.as_retriever(
            search_type="similarity",
            search_kwargs={"k": CHROMA_VECTOR_RESULT_CNT}
        )

    def _get_semantic_storage(self) -> Chroma:
        """Instantiates the isolated semantic database research partition."""
        return Chroma(
            client=self.chromadb_client,
            embedding_function=self.embedding_model,
            collection_name="semantic_chunks"
        )

    def _get_vector_storage(self) -> Chroma:
        """Instantiates the generalized primary target layout collection."""
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
            breakpoint_threshold_amount=SEMANTIC_THRESH_LIMIT
        )

    def _get_structured_retriever(self) -> SelfQueryRetriever:
        return SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.semantic_storage,
            document_contents="Text Semantic Chunks for " + DOCUMENT_FILE + " published by the Global Nutritional Health Organization",
            metadata_field_info=[
                AttributeInfo(
                    name="source",
                    description=f"The file path or name of the medical reference document (e.g., '{DOCUMENT_FILE}')",
                    type="string",
                ),
                AttributeInfo(
                    name="page",
                    # description="The explicit page number within the parsed medical document (0-indexed)",
                    description="The page number in the PDF document, starting from 0. Use integers for comparisons.",
                    type="integer",
                ),

                AttributeInfo(
                    name="page_content",
                    description="raw text of (sectional) document",
                    type="string"
                )
            ],
            structured_query_translator=ChromaTranslator(),
            verbose=True,
            use_original_query=True,
        )

    def _get_structured_hyp_retriever(self) -> SelfQueryRetriever:

        return SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vector_storage,
            document_contents="Hypothetical Questions for " + DOCUMENT_FILEPATH + " published by the Global Nutritional Health Organization",
            metadata_field_info=[
                AttributeInfo(
                    name="original_content",
                    description="Original text extracted from documents",
                    type="string"
                ),
                AttributeInfo(
                    name="source",
                    description="File path of document",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="Page number of document",
                    type="integer"
                ),
                AttributeInfo(
                    name="type",
                    description="Datatype of attribute `original_content`",
                    type="string"
                )
            ],
            structured_query_translator=ChromaTranslator(),
            verbose=True,
            use_original_query=True
        )

    def get_semantic_chunks(self, folder_path: str) -> list:
        semantic_chunks = []
        pdf_loader = PyPDFDirectoryLoader(folder_path)
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
        print('# --- Adding semantic documents --- #')
        batch_size = DOCUMENT_CHUNK_BATCH_SIZE #100
        for i in range(0, len(semantic_chunks), batch_size):
            self.semantic_storage.add_documents(semantic_chunks[i: i + batch_size])

    def add_vector_documents(self, documents: list) -> None:
        print('# --- Adding vector documents --- #')
        batch_size = DOCUMENT_CHUNK_BATCH_SIZE
        for i in range(0, len(documents), batch_size):
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
            #'document_content_description': self.document_content_description,
            'embedding_model': self.embedding_model,
            'force_rebuild': self.force_rebuild,
            'llm': self.llm,
            #'metadata_info': self.metadata_info,
        }

    def query_questions(self, is_hyp: bool = False, pluck: bool = False) -> None:
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
                print(f"\n----- {I_QUES}Question #{i+1} {I_QUES} -----")
                print(question)
                print(f"\n{I_INFO} Number of Semantic Chunks Retrieved: {retrieved_count}")
                
                # Display empty retrieval
                if retrieved_count == 0:
                    print(f'{I_WARNING}  Nothing was retrieved! {I_WARNING}')
                else:
                    print(f"{I_DOCUMENT} Retrieved Documents:\n{ques_semantic_chunks_retrieved}")
                
                results_count.append(retrieved_count)

            except Exception as e:
                print(f"⚠️ Skipping query due to parser error: {question}.")
                print(f"\tReason: {e}")
                results_count.append(0)

            print(f"+---- Question #{i+1} ----+\n")
            time.sleep(SLEEP_TIME_SEC)

        total = len(queries_to_run)
        successful = sum(1 for c in results_count if c > 0)
        print(f"Retriever Quality: {successful}/{total} queries returned results ({successful/total*100:.0f}%)")

    @staticmethod
    # @todo - should this be moved to config.py?
    def get_new_sleep_time() -> float:
        # Used in child classes.
        return RATE_LIMIT_TIME + random.uniform(2.0, 7.0)