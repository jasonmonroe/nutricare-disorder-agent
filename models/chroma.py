# models/chroma.py

import logging
import os
import random
import chromadb

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma  
# 👑 CHANGED: Import the explicit local huggingface transformer 
# to keep embeddings separate from Groq API Base routing loops
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker

from src.config import (
    CHROMA_SERVER_NO_TELEMETRY,
    DOCUMENT_CHUNK_BATCH_SIZE,
    DOCUMENT_DIR,
    I_QUES,
    OPENAI_API_BASE, 
    OPENAI_API_KEY,
    SEMANTIC_THRESH_LIMIT,
    SIMILARITY_SEARCH_QUERY,
    VECTOR_RESULT_CNT,
    VECTORS_DIR
)

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

        self.collection_name = None
        self.document_content_description = None
        
        # 👑 DYNAMIC DECOUPLING: Instantiate a local embedding layer 
        # that bypasses OpenAI / Groq network proxy formatting entirely
        self.embedding_model = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
        
        self.llm = None
        self.metadata_info = {}

        # Set incoming pipeline dictionary variables
        self._set_attrs(dataset)

        # Build downstream storage partitions
        self.semantic_storage = self._get_semantic_storage()
        self.vector_storage = self._get_vector_storage()
        self.retriever = self.get_retriever()
        
        self.semantic_text_splitter = self._get_semantic_text_splitter()

        # Build self-query capabilities
        self.structured_retriever = self._get_structured_retriever()
        self.structured_hyp_retriever = self._get_structured_hyp_retriever()

    def _set_attrs(self, dataset: dict) -> None:
        for key, value in dataset.items():
            if hasattr(self, key) and key != "embedding_model": # Keep our local embedding object protected
                setattr(self, key, value)

        if self.llm is None and 'openai_model' in dataset:
            openai_model = dataset['openai_model']
            self.llm = openai_model.llm

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
        # 👑 FIX: Removed persist_directory to respect the master path defined in self.chromadb_client.
        return Chroma(
            client=self.chromadb_client,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name
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
            document_contents="Text Semantic Chunks for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization",
            metadata_field_info=[
                AttributeInfo(name="page", description="page number of document", type="integer"),
                AttributeInfo(name="source", description="file path of document", type="string"),
                AttributeInfo(name="page_content", description="raw text of (sectional) document", type="string")
            ],
            verbose=True
        )

    def _get_structured_hyp_retriever(self) -> SelfQueryRetriever:
        return SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vector_storage,
            document_contents="Hypothetical Questions for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization",
            metadata_field_info=[
                AttributeInfo(name="original_content", description="Original text extracted from documents", type="string"),
                AttributeInfo(name="source", description="File path of document", type="string"),
                AttributeInfo(name="page", description="Page number of document", type="integer"),
                AttributeInfo(name="type", description="Datatype of attribute `original_content`", type="string")
            ],
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

    def query_questions(self, is_hyp: bool = False, pluck: bool = False) -> None:
        """
        Query questions using either structured hypothetical retriever or structured retriever.
        :param is_hyp:
        :param pluck:
        :return: None
        """
        retriever = self.structured_hyp_retriever if is_hyp else self.structured_retriever
        print('--- Hypothetical Retriever ---' if is_hyp else '--- Retriever ---')

        if pluck:
            ques = random.choice(self.queries())
            semantic_chunks_retrieved = retriever.invoke(ques)
            print(f"Question: {ques}{I_QUES}")
            print(f"Retrieved Documents: {semantic_chunks_retrieved}")
        else:
            for ques in self.queries():
                semantic_chunks_retrieved = retriever.invoke(ques)
                print(f"Number of Semantic Chunks Retrieved: {len(semantic_chunks_retrieved)}")
                print(f"Question: {ques}{I_QUES}")
                print(f"Retrieved Documents: {semantic_chunks_retrieved}")
                print("---\n")

    @staticmethod
    def queries() -> list:
        return [
            "What is the recommended dosage for treating scurvy?",
            "What is definition of Vitamin C but only from content found in the first quarter of the document?",
            "Describe the clinical signs of deficiency but exclude any data from the source `Pediatric Nutrition Guide.pdf`.",
            "What type of nutritional support is needed for patients to increase lean body mass?",
            "On page 35, please explain the correlation between Vitamin B12 levels and tissue deficiency.",
            "What are the laboratory and clinical standards for diagnosing Vitamin D deficiency? Specifically address adult patients."
        ]
        