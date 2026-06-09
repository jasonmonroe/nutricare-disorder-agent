# models/chroma.py

# +----------------+
# |     CHROMA     |
# +----------------+

# Python Libraries
import logging
import os
import random

# Vendor Libraries
import chromadb

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma  
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
# Core asset loading layers maintained inside generic community spaces
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
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

        # Initialize ChromaDB client without diagnostic telemetry overheads
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = CHROMA_SERVER_NO_TELEMETRY
        os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)


        # --- Temp --- #
        # This prevents the 1032 Readonly Database lock when running consecutive pipelines.
        if self._mock:
            try:
                from chromadb.api.shared_system_client import SharedSystemClient
                SharedSystemClient.clear()
            except Exception:
                pass
        # --- Temp --- #

        self.chromadb_client = chromadb.PersistentClient(path=f"./{VECTORS_DIR}")

        self.collection_name = None
        self.document_content_description = None
        self.embedding_model = None
        self.llm = None
        self.metadata_info = {}

        # Dynamically set class attributes based on the incoming configuration state
        self._set_attrs(dataset)

        self.retriever = self.get_retriever()
        self.semantic_text_splitter = self._get_semantic_text_splitter()
        self.semantic_storage = self._get_semantic_storage()
        self.vector_storage = self._get_vector_storage()

        # Self-Querying layers initialized last after underlying Chroma collections are ready
        self.structured_retriever = self._get_structured_retriever()
        self.structured_hyp_retriever = self._get_structured_hyp_retriever()

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

    def _set_attrs(self, dataset: dict) -> None:
        """
        Updates class attributes dynamically. If llm isn't set, but we have openai model in the dataset set that attr.
        :param dataset:
        :return: None
        """

        for key, value in dataset.items():
            if hasattr(self, key):
                #print(f'\nSetting {key} = {value}')
                setattr(self, key, value)

        if self.llm is None and 'openai_model' in dataset:
            openai_model = dataset['openai_model']
            self.llm = openai_model.llm
            #print(f'\n\nsetting self.llm = openai_model.llm')

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
        :return: None
        """

        retriever = self.structured_hyp_retriever if is_hyp else self.structured_retriever
        print('\n--- Hypothetical Retriever ---' if is_hyp else '\n--- Retriever ---')

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

    def _get_semantic_text_splitter(self) -> SemanticChunker:
        """Initializes the semantic text splitter using the target embedding framework."""

        """
        Initializes a local text splitter to avoid heavy proxy batch overheads.
        Note: ⚠️ Use RecursiveCharacterTextSplitter() until credentials are validated.
        """
        # --- Temp --- #
        if self._mock:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter(
                chunk_size=1000,       # Adjust based on your preferred configuration size
                chunk_overlap=200,
                length_function=len
            )
            # --- Temp --- #

        return SemanticChunker(
            self.embedding_model,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=SEMANTIC_THRESH_LIMIT
        )

    def get_retriever(self) -> VectorStoreRetriever:
        """Initializes vector retriever and gets vectorized data stored in Chroma."""
        vector_storage = Chroma(
            client=self.chromadb_client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            #persist_directory=f"{self.collection_name}_db"
            persist_directory=self._format_dir(self.collection_name)
        )

        return vector_storage.as_retriever(
            search_type="similarity",
            search_kwargs={"k": VECTOR_RESULT_CNT}
        )

    def _get_structured_retriever(self) -> SelfQueryRetriever:
        """Creates LangChain Structured Self-Query Retriever for research chunks."""
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
        """Creates LangChain Structured Self-Query Retriever for hypothetical queries."""
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
        """
        Gets semantic chunks from documents located within the target directory path.
        :param folder_path:
        :return: list
        """

        semantic_chunks = []
        pdf_loader = PyPDFDirectoryLoader(folder_path)
        chunks = pdf_loader.load_and_split(self.semantic_text_splitter)
        semantic_chunks.extend(chunks)
        print(f"Total Semantic Chunks Created: {len(semantic_chunks)}")
        
        return semantic_chunks

    def _format_dir(self, path: str) -> str:
        """
        Formats the unified persistence destination naming template.
        :param path:
        :return:
        """

        return f"./{VECTORS_DIR}/{path}_db"

    def _get_semantic_storage(self) -> Chroma:
        """Instantiates the specialized semantic storage partition layer."""
        return Chroma(
            client=self.chromadb_client,
            embedding_function=self.embedding_model,
            collection_name="semantic_chunks",
            #persist_directory=self._format_dir("research")
        )

    def add_semantic_documents(self, semantic_chunks: list) -> None:
        """
        Batches and commits document nodes out to the semantic database index.
        :param semantic_chunks:
        :return:
        """

        batch_size = DOCUMENT_CHUNK_BATCH_SIZE
        for i in range(0, len(semantic_chunks), batch_size):
            batch = semantic_chunks[i: i + batch_size]
            self.semantic_storage.add_documents(batch)

    def _get_vector_storage(self) -> Chroma:
        """Instantiates the generalized target collection vector layer."""
        return Chroma(
            client=self.chromadb_client,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            #persist_directory=self._format_dir(self.collection_name)
        )

    def add_vector_documents(self, documents: list) -> None:
        """
        Batches and commits complex vector entities out to storage collection space.
        :param documents:
        :return:
        """

        batch_size = DOCUMENT_CHUNK_BATCH_SIZE
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.vector_storage.add_documents(batch)

    def get_documents(self) -> list:
        # Returns documents used by the doc_handler.
        return self.semantic_storage.similarity_search(
            query=SIMILARITY_SEARCH_QUERY,
            k=VECTOR_RESULT_CNT
        )