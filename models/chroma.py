# models/chroma.py

import logging
import os
import random

# Vendor Libraries
import chromadb


# --- FIXED MODERN LANGCHAIN CORE & EXTENSION IMPORTS ---
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma  
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# The canonical, native namespace for Self-Query schema tokens in LangChain v1.x:
#from langchain_core.structured_query import AttributeInfo  
from langchain.chains.query_constructor.schema import AttributeInfo

# Self-Query and Document Compression routing via standardized engine namespaces
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


# FIXED: Rerankers are now imported from the community collection explicitly
from langchain_community.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# Core asset loading layers maintained inside generic community spaces
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from src.config import (
    DOCUMENT_CHUNK_BATCH_SIZE,
    DOCUMENT_DIR,
    I_QUES,
    SEMANTIC_THRESH_LIMIT,
    VECTOR_RESULT_CNT,
    VECTORS_DIR,
)


class ChromaModel:

    # https://www.trychroma.com
    # Documentation https://docs.trychroma.com/docs/overview/introduction
    # Documentation https://docs.trychroma.com/docs/querying-collections/query-and-get

    def __init__(self, dataset: dict):

        # Initialize ChromaDB client
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = "true"
        logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

        self.chromadb_client = chromadb.PersistentClient()

        self.collection_name = None
        self.document_content_description = None
        self.embedding_model = None
        self.llm = None
        self.metadata_info = {}

        # Set all attributes to what's in the dataset.
        self._set_attrs(dataset)

        self.retriever = self.get_retriever()
        self.semantic_text_splitter = self._get_semantic_text_splitter()
        self.semantic_storage = self._get_semantic_storage()
        self.vector_storage = self._get_vector_storage()

        # Self Querying layers initialized last after the underlying Chroma backends are ready
        self.structured_retriever = self._get_structured_retriever()
        self.structured_hyp_retriever = self._get_structured_hyp_retriever()

    @staticmethod
    def queries() -> list:
        return [
            "What is the recommended dosage for scurvy?",
            "What is definition of Vitamin C but only from content found in the first quarter of the document?",
            "Describe the clinical signs of deficiency but exclude any data from the source `Pediatric Nutrition Guide.pdf`.",
            "What type of nutritional support is needed for patients to increase lean body mass?",
            "On page 35, please explain the correlation between Vitamin B12 levels and tissue deficiency.",
            "What are the laboratory and clinical standards for diagnosing Vitamin D deficiency? Specifically address adult patients."
        ]

    def _set_attrs(self, dataset: dict) -> None:
        """
        Update class attributes dynamically.
        
        :param dataset
        :return None
        """
        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.llm is None and 'openai_model' in dataset:
            openai_model = dataset['openai_model']
            self.llm = openai_model.llm

    def export(self) -> dict:
        """
        Exports attributes to be injected into child classes
        :return:
        """
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
        :param is_hyp: is retriever hypothetical or not?
        :param pluck: do we query all questions or pluck a random to test the retriever?
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

    def _get_semantic_text_splitter(self):
        """
        Initializes the semantic text splitter, controlling how the text is divided into meaningful chunks.
     
        :return: SemanticChunker
        """
        return SemanticChunker(
            self.embedding_model,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=SEMANTIC_THRESH_LIMIT
        )

    def get_retriever(self) -> VectorStoreRetriever:
        """
        Initializes vector retriever and gets vectorized data stored in Chroma
        The `persist directory` is from the root repository path, not the Google Colab path.
        :return: VectorStoreRetriever
        """
        vector_storage = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=f"{self.collection_name}_db"
        )

        return vector_storage.as_retriever(
            search_type="similarity",
            search_kwargs={"k": VECTOR_RESULT_CNT}
        )

    def _get_structured_retriever(self):
        """
        Creates LangChain Structured Receiver
        :return: SelfQueryRetriever
        """
        return SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.semantic_storage,
            document_contents="Text Semantic Chunks for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization",
            metadata_field_info=[
                AttributeInfo(
                    name="page",
                    description="page number of document",
                    type="integer"
                ),
                AttributeInfo(
                    name="source",
                    description="file path of document",
                    type="string"
                ),
                AttributeInfo(
                    name="page_content",
                    description="raw text of (sectional) document",
                    type="string"
                )
            ],
            verbose=True
        )

    def _get_structured_hyp_retriever(self):
        """
        Creates LangChain Structured Receiver
        :return: SelfQueryRetriever
        """
        return SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vector_storage,
            document_contents="Hypothetical Questions for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization",
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
            verbose=True
        )

    def get_semantic_chunks(self, folder_path) -> list:
        """
        Gets semantic chunks from directory.
        Note: This method is called outside this class.
        :param folder_path:
        :return:
        """
        semantic_chunks = []
        pdf_loader = PyPDFDirectoryLoader(folder_path)
        chunks = pdf_loader.load_and_split(self.semantic_text_splitter)
        semantic_chunks.extend(chunks)
        print(f"Total Semantic Chunks Created: {len(semantic_chunks)}")
        return semantic_chunks

    def _format_dir(self, path: str) -> str:
        """
        :param path: path of directory
        :return: full string of directory path formatted
        """
        return f"./{VECTORS_DIR}/{path}_db"

    def _get_semantic_storage(self) -> Chroma:
        return Chroma(
            embedding_function=self.embedding_model,
            collection_name="semantic_chunks",
            persist_directory=self._format_dir("research")
        )

    def add_semantic_documents(self, semantic_chunks: list) -> None:
        """
        Note: called outside the class
        :param semantic_chunks:
        :return: None
        """
        batch_size = DOCUMENT_CHUNK_BATCH_SIZE
        for i in range(0, len(semantic_chunks), batch_size):
            batch = semantic_chunks[i: i + batch_size]
            self.semantic_storage.add_documents(batch)

    def _get_vector_storage(self) -> Chroma:
        return Chroma(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            persist_directory=self._format_dir(self.collection_name)
        )

    def add_vector_documents(self, documents) -> None:
        batch_size = DOCUMENT_CHUNK_BATCH_SIZE
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.vector_storage.add_documents(batch)
