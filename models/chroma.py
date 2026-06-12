# models/chroma.py

# Python Libraries
import logging
import os
import random
import chromadb

# Vector Libraries
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.query_constructors.chroma import ChromaTranslator

from src.config import (
    CHROMA_SERVER_NO_TELEMETRY,
    DOCUMENT_CHUNK_BATCH_SIZE,
    DOCUMENT_FILE,
    I_INFO,
    I_QUES,
    RATE_LIMIT_TIME,
    SEMANTIC_THRESH_LIMIT,
    SIMILARITY_SEARCH_QUERY,
    VECTOR_RESULT_CNT,
    VECTORS_DIR, DOCUMENT_FILEPATH,
)

from src.utils import format_dir

class ChromaModel:
    """
    Manages the persistent vector store lifecycle using ChromaDB and handles both similarity-based and
    metadata-structured Self-Query retrieval mechanisms.
    """

    def __init__(self, dataset: dict):
        self._mock = dataset.get('mock', False)

        # Force telemetry down
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = CHROMA_SERVER_NO_TELEMETRY
        logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

        # Clear system client cache if running a mock pipeline test
        if self._mock:
            try:
                from chromadb.api.shared_system_client import SharedSystemClient
                SharedSystemClient.clear()
            except Exception:
                pass

        # 👑 LOCK PATH HERE: This client dictates exactly where data lands on disk
        # Ensures everything stays tightly isolated inside your db directory
        self.chromadb_client = chromadb.PersistentClient(path=os.path.abspath(VECTORS_DIR))

        self.collection_name = ''
        self.document_content_description = ''
        self.embedding_model = None 
        self.llm = None
        self.metadata_info = {}

        # Set incoming pipeline dictionary variables
        self._set_attrs(dataset)

        # Build downstream storage partitions
        self.semantic_storage = self._get_semantic_storage()
        self.vector_storage = self._get_vector_storage()
        self.retriever = self.get_retriever() if not self._mock else None
        self.semantic_text_splitter = self._get_semantic_text_splitter() if not self._mock else None
        self.structured_retriever = self._get_structured_retriever() if not self._mock else None
        self.structured_hyp_retriever = self._get_structured_hyp_retriever() if not self._mock else None

    def _set_attrs(self, dataset: dict) -> None:
        """Safely maps dataset keys to class attributes, avoiding method overwrites."""
        # Whitelist of allowed attributes to prevent overwriting internal methods or private variables
        #target_attrs = {'collection_name', 'document_content_description', 'embedding_model', 'llm', 'metadata_info', 'mock'}
        
        for key, value in dataset.items():
            if hasattr(self, key):
                print(f'DEBUG: key={key}, value={value}')
                setattr(self, key, value)

        if self.llm is None and 'openai_model' in dataset:
            openai_model = dataset['openai_model']
            self.llm = openai_model.llm
            print(f'\nself.llm = {self.llm}')

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
        # 👑 FIX: Removed persist_directory parameter. 
        # The underlying 'client' object safely handles the persistent directory paths now.
        return self.vector_storage.as_retriever(
            search_type="similarity",
            search_kwargs={"k": VECTOR_RESULT_CNT}
        )

    def _get_semantic_storage(self) -> Chroma:
        """Instantiates the isolated semantic database research partition."""
        # 👑 FIX: Removed persist_directory to prevent duplicate folder creation in root.
        return Chroma(
            client=self.chromadb_client,
            embedding_function=self.embedding_model,
            collection_name="semantic_chunks"
        )

    def _get_vector_storage(self) -> Chroma:
        """Instantiates the generalized primary target layout collection."""
        # 👑 @todo FIX: Removed persist_directory to respect the master path defined in self.chromadb_client.
        return Chroma(
            client=self.chromadb_client,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            #persist_directory=format_dir(self.collection_name)
        )

    def _get_semantic_text_splitter(self) -> SemanticChunker:
        if self._mock:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

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
            metadata_field_info = [
                AttributeInfo(
                    name="source",
                    description=f"The file path or name of the medical reference document (e.g., '{DOCUMENT_FILE}')",
                    type="string",
                ),
                AttributeInfo(
                    name="page",
                    description="The explicit page number within the parsed medical document (0-indexed)",
                    type="integer",
                ),
                AttributeInfo(
                    name="filename",
                    description=f"The bare filename of the medical reference document (e.g., '{DOCUMENT_FILE}')",
                    type="string",
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
            document_contents="Questions for " + DOCUMENT_FILE + " published by the Global Nutritional Health Organization",
            metadata_field_info=[
                AttributeInfo(
                    name="original_content", 
                    description="Original text extracted from medical documents", 
                    type="string"
                ),
                AttributeInfo(
                    name="source", 
                    description="File path or name of document", 
                    type="string"
                ),
                AttributeInfo(
                    name="page", 
                    description="Page number of document", 
                    type="integer"
                ),
                AttributeInfo(
                    name="type", 
                    description="Content type (e.g., 'table', 'text')", 
                    type="string"
                )
            ],
            structured_query_translator=ChromaTranslator(),
            verbose=True
        )

    def get_semantic_chunks(self, folder_path: str) -> list:
        semantic_chunks = []
        pdf_loader = PyPDFDirectoryLoader(folder_path)
        chunks = pdf_loader.load_and_split(self.semantic_text_splitter)
        semantic_chunks.extend(chunks)

        return semantic_chunks

    def add_semantic_documents(self, semantic_chunks: list) -> None:
        batch_size = DOCUMENT_CHUNK_BATCH_SIZE
        for i in range(0, len(semantic_chunks), batch_size):
            self.semantic_storage.add_documents(semantic_chunks[i: i + batch_size])

    def add_vector_documents(self, documents: list) -> None:
        batch_size = DOCUMENT_CHUNK_BATCH_SIZE
        for i in range(0, len(documents), batch_size):
            self.vector_storage.add_documents(documents[i : i + batch_size])

    def get_documents(self) -> list:
        return self.semantic_storage.similarity_search(
            query=SIMILARITY_SEARCH_QUERY,
            k=VECTOR_RESULT_CNT
        )

    def get_document_count(self) -> int:
        """Returns the number of documents currently in this collection."""
        try:
            return self.vector_storage._collection.count()
        except Exception:
            return 0

    def get_semantic_count(self) -> int:
        """Returns the number of documents in the semantic storage collection."""
        try:
            return self.semantic_storage._collection.count()
        except Exception:
            return 0


    def export(self) -> dict:
        """Exports attributes to be injected into child or dependent pipeline classes."""
        return {
            'collection_name': self.collection_name,
            'document_content_description': self.document_content_description,
            'embedding_model': self.embedding_model,
            'llm': self.llm,
            'metadata_info': self.metadata_info,
        }


    def query_questions(self, is_hyp: bool = False, pluck: bool = False) -> None:
        """
        Query questions using either structured hypothetical retriever or structured retriever.

        :param is_hyp:
        :param pluck:
        :return:
        """

        retriever = self.structured_hyp_retriever if is_hyp else self.structured_retriever
        print('--- Hypothetical Retriever ---' if is_hyp else '--- Retriever ---')

        queries_to_run = [random.choice(self.queries())] if pluck else self.queries()
        results_count = []

        for question in queries_to_run:
            try:
                ques_semantic_chunks_retrieved = retriever.invoke(question)
                print(f"{I_INFO} Number of Semantic Chunks Retrieved: {len(ques_semantic_chunks_retrieved)}")
                print(f"Question: {question}{I_QUES}")
                print(f"{I_INFO} Retrieved Documents: {ques_semantic_chunks_retrieved}")
                results_count.append(len(ques_semantic_chunks_retrieved))

            except Exception as e:
                print(f"⚠️ Skipping query due to parser error: {question}.")
                print(f"\tReason: {e}")
                results_count.append(0)

            print("---\n")

        total = len(queries_to_run)
        successful = sum(1 for c in results_count if c > 0)
        print(f"Retriever Quality: {successful}/{total} queries returned results ({successful/total*100:.0f}%)")

    @staticmethod
    def get_new_sleep_time() -> float:
        return RATE_LIMIT_TIME + random.uniform(2.0, 7.0)