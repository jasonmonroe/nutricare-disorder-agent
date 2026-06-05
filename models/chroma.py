# models/chroma.py


import logging
import os
import random

# Vendors

#import chromadb  # High-performance vector database for storing/querying dense vectors
import chromadb
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_community.chat_models.openai import ChatOpenAI

# LangChain community & experimental imports
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader  # Document loaders for PDFs
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers.self_query.base import SelfQueryRetriever  # Base classes for self-querying retrievers

from langchain_openai import OpenAIEmbeddings
from src.config import (
    SEMANTIC_THRESH_LIMIT, VECTOR_COLLECTION_NAME, VECTOR_RESULT_CNT, VECTORS_DIR, I_QUES, DOCUMENT_CHUNK_BATCH_SIZE,
    DOCUMENT_DIR
)


# Implementations of vector stores like Chroma


class ChromaModel:

    # https://www.trychroma.com
    # Documentation https://docs.trychroma.com/docs/overview/introduction
    # Documentation https://docs.trychroma.com/docs/querying-collections/query-and-get

    def __init__(self, dataset: dict):

        # Initialize ChromaDB client
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = "true"
        logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

        self.chromadb_client = chromadb.EphemeralClient()
        self.collection_name = dataset['collection_name']
        self.document_content_description = ''
        #self.metadata = {}
        self.metadata_info = {}

        # --- INITIALIZE CHROMA VECTOR STORAGE FOR RETRIEVING DOCUMENTS --- #
        # Retrieve `nutritional` database created from Google Colab
        self.retriever = self.get_retriever(self.collection_name, dataset['embedding_model'])
        self.structured_retriever = self._get_structured_retriever(dataset['llm'])
        self.structured_hyp_retriever = self._get_structured_hype_retriever(dataset['llm'])
        self.semantic_text_splitter = self._get_semantic_text_splitter(dataset['embedding_model'])
        self.semantic_storage = self._get_semantic_storage(dataset['embedding_model'])
        self.vector_storage = self._get_vector_storage(dataset['embedding_model'])

        self._set_attrs(dataset)

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
        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def processed_data(self) -> dict:
        # Data stored from parent class that's returned to each child/grand child class
        return {}

    def query_questions(self, is_hyp: bool=False, pluck=False) -> None:

        if is_hyp:
            retriever = self.structured_hyp_retriever
            print('--- Hypothetical Retriever ---')
        else:
            retriever = self.structured_retriever
            print('--- Retriever ---')

        # Only invoke a random question plucked from the list of queries
        if pluck:

            ques_idx = random.choice(self.queries())
            ques = self.queries()[ques_idx]
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




    def _get_semantic_text_splitter(self, embedding_model):
        # This initializes the semantic text splitter, controlling how the text is divided into meaningful chunks.
        return SemanticChunker(
            embedding_model,  # Fill in the embedding model
            breakpoint_threshold_type='percentile',  # Choose the threshold type (e.g., 'percentile')
            breakpoint_threshold_amount=SEMANTIC_THRESH_LIMIT  # Set the chunking threshold (e.g., 80, 85)
        )

    # Initializes vector retriever and gets vectorized data stored in Chroma
    # The `persist directory` is from the root repository path, not the Google Colab path.
    def get_retriever(self, embedding_model: OpenAIEmbeddings, collection_name: str):
        vector_storage = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=f"{collection_name}_db"
        )

        # Create a retriever from the vector store
        return vector_storage.as_retriever(
            search_type="similarity",
            search_kwargs={"k": VECTOR_RESULT_CNT}
        )

    def _get_structured_retriever(self, llm: ChatOpenAI ) -> SelfQueryRetriever:
        return SelfQueryRetriever.from_llm(
            llm,
            self.semantic_storage,
            "Text Semantic Chunks for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization",
            [
                AttributeInfo(
                    name="page",  # Fill in the metadata field name (e.g., "Category")
                    description="page number of document",  # Describe what this field represents
                    type="integer"  # Fill in the data type (e.g., "string", "integer")
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

    def _get_structured_hyp_retriever(self, llm: ChatOpenAI):
        return SelfQueryRetriever.from_llm(
            llm,                           # LLM model
            self.vector_storage,                   # Vectorstore
            "Hypothetical Questions for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization",
            [
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
            ]        # Metadata field info
        )

    def get_semantic_chunks(self, folder_path) -> list:
        semantic_chunks = []

        # Step 3: Initialize the PyPDFDirectoryLoader for the folder
        pdf_loader = PyPDFDirectoryLoader(folder_path)  # Use the correct loader (e.g., PyPDFDirectoryLoader)

        # Step 4: Load and split PDF documents into chunks using SemanticChunker
        chunks = pdf_loader.load_and_split(self.semantic_text_splitter)  # Call the appropriate function and pass the splitter

        # Step 5: Extend the semantic_chunks list with the chunks from this folder
        semantic_chunks.extend(chunks)  # Add chunks to the list

        # Step 6: Get the total number of chunks
        print(f"Total Semantic Chunks Created: {len(semantic_chunks)}")

        return semantic_chunks

    # Format persist
    # directory
    def _format_dir(self, path: str) -> str:
        return f"./{VECTORS_DIR}/{path}_db"

    def _get_semantic_storage(self, embedding_model: OpenAIEmbeddings) -> Chroma:
        return Chroma(
            embedding_function=embedding_model,
            collection_name="semantic_chunks",
            persist_directory=self._format_dir("research")
        )

    def add_semantic_documents(self, semantic_chunks: dict):
        batch_size = DOCUMENT_CHUNK_BATCH_SIZE  # Adjust the batch size as needed
        for i in range(0, len(semantic_chunks), batch_size):
            batch = semantic_chunks[i: i + batch_size]
            self.semantic_storage.add_documents(batch)

    def _get_vector_storage(self, embedding_model: OpenAIEmbeddings):
        return Chroma(
            embedding_function=embedding_model,
            collection_name=self.collection_name,
            persist_directory=self._format_dir(self.collection_name)
        )

    def add_vector_documents(self, documents):
        # Note: documents are either  hypothetical_questions or table_hypothetical questions
        batch_size = DOCUMENT_CHUNK_BATCH_SIZE
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.vector_storage.add_documents(batch)




    #def get_hypothetical_questions(self):
    #    pass
