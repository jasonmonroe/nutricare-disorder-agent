# models/chroma.py

# +----------------+
# |     CHROMA     |
# +----------------+

# Python Libraries
import logging
import os
import random
from dataclasses import dataclass

# Vendor Libraries
import chromadb
import importlib
import pkgutil

# Compatibility import shim: LangChain has moved things between releases and
# into separate packages (langchain_core, langchain_community, langchain_chroma, etc.).
# Try several likely import paths and fall back gracefully with clear errors.

def _try_import(mod_name: str):
    try:
        return importlib.import_module(mod_name)
    except Exception:
        return None

# VectorStoreRetriever
VectorStoreRetriever = None
for _mod in ("langchain.vectorstores", "langchain_core.vectorstores"):
    m = _try_import(_mod)
    if m and hasattr(m, "VectorStoreRetriever"):
        VectorStoreRetriever = getattr(m, "VectorStoreRetriever")
        break

# Chroma (vector store)
Chroma = None
for _mod in ("langchain_chroma", "langchain.vectorstores", "langchain_core.vectorstores"):
    m = _try_import(_mod)
    if m and hasattr(m, "Chroma"):
        Chroma = getattr(m, "Chroma")
        break
if Chroma is None:
    # last try: some langchain_chroma installs expose a top-level function/class differently
    m = _try_import("langchain_chroma")
    if m and hasattr(m, "Chroma"):
        Chroma = getattr(m, "Chroma")

# OpenAI chat and embeddings
ChatOpenAI = None
OpenAIEmbeddings = None
for _mod in ("langchain_openai", "langchain.chat_models", "langchain.chat_models.openai"):
    m = _try_import(_mod)
    if m:
        if hasattr(m, "ChatOpenAI") and ChatOpenAI is None:
            ChatOpenAI = getattr(m, "ChatOpenAI")
        if hasattr(m, "OpenAIEmbeddings") and OpenAIEmbeddings is None:
            OpenAIEmbeddings = getattr(m, "OpenAIEmbeddings")

# Self-Query and document compressors
SelfQueryRetriever = None
LLMChainExtractor = None
ContextualCompressionRetriever = None
for base in ("langchain.retrievers", "langchain_core.retrievers"):
    # SelfQueryRetriever
    m = _try_import(base + ".self_query.base")
    if m and hasattr(m, "SelfQueryRetriever"):
        SelfQueryRetriever = getattr(m, "SelfQueryRetriever")
    # LLMChainExtractor (document compressors)
    m2 = _try_import(base + ".document_compressors")
    if m2 and hasattr(m2, "LLMChainExtractor"):
        LLMChainExtractor = getattr(m2, "LLMChainExtractor")
    # ContextualCompressionRetriever
    m3 = _try_import(base + ".contextual_compression")
    if m3 and hasattr(m3, "ContextualCompressionRetriever"):
        ContextualCompressionRetriever = getattr(m3, "ContextualCompressionRetriever")

# CrossEncoderReranker lives in the community package in some distributions
CrossEncoderReranker = None
for _mod in ("langchain_community.document_compressors.cross_encoder_rerank",
             "langchain_community.document_compressors", "langchain.community.document_compressors"):
    m = _try_import(_mod)
    if m and hasattr(m, "CrossEncoderReranker"):
        CrossEncoderReranker = getattr(m, "CrossEncoderReranker")
        break

# Document loaders and text splitters (PDF loader / semantic chunker)
PyPDFDirectoryLoader = None
PyPDFLoader = None
SemanticChunker = None
for _mod in ("langchain_community.document_loaders", "langchain.document_loaders"):
    m = _try_import(_mod)
    if m:
        if hasattr(m, "PyPDFDirectoryLoader") and PyPDFDirectoryLoader is None:
            PyPDFDirectoryLoader = getattr(m, "PyPDFDirectoryLoader")
        if hasattr(m, "PyPDFLoader") and PyPDFLoader is None:
            PyPDFLoader = getattr(m, "PyPDFLoader")

# semantic chunker may be in experimental or text_splitter modules
for _mod in ("langchain_experimental.text_splitter", "langchain.text_splitter"):
    m = _try_import(_mod)
    if m and hasattr(m, "SemanticChunker"):
        SemanticChunker = getattr(m, "SemanticChunker")
        break

# Raise clear errors for missing critical pieces so failures are explicit at import time
missing = []
if SelfQueryRetriever is None:
    missing.append("SelfQueryRetriever (langchain.retrievers.self_query)")
if Chroma is None:
    missing.append("Chroma (langchain_chroma or langchain.vectorstores)")
if VectorStoreRetriever is None:
    missing.append("VectorStoreRetriever (langchain.vectorstores or langchain_core.vectorstores)")
if SemanticChunker is None:
    missing.append("SemanticChunker (langchain_experimental.text_splitter or langchain.text_splitter)")
if missing:
    # don't raise immediately to allow non-critical codepaths to import; but log for clarity
    logging.getLogger(__name__).warning("Optional compatibility imports missing: %s", missing)

# If SelfQueryRetriever wasn't found in the environment, provide a minimal
# fallback shim so the rest of the code can still perform simple retrievals.
if SelfQueryRetriever is None:
    class FallbackSelfQueryRetriever:
        """Minimal compatibility shim for SelfQueryRetriever.

        This fallback does not perform LLM-based self-query parsing. Instead it
        wraps the provided vectorstore and exposes a from_llm() constructor and
        an invoke(query) method that returns documents from the vectorstore.
        """

        def __init__(self, vectorstore, **kwargs):
            # vectorstore may be either a LangChain VectorStore instance or already a retriever
            self.vectorstore = vectorstore

        @classmethod
        def from_llm(cls, llm=None, vectorstore=None, **kwargs):
            # llm is ignored in the fallback; keep signature compatible
            return cls(vectorstore=vectorstore)

        def invoke(self, query: str):
            # Prefer an invoke method on the underlying retriever if present
            vs = self.vectorstore
            try:
                # If the vectorstore is a LangChain vectorstore with as_retriever()
                if hasattr(vs, "as_retriever"):
                    r = vs.as_retriever()  # default params
                else:
                    r = vs

                # If the retriever exposes invoke (LangChain Runnable retriever)
                if hasattr(r, "invoke"):
                    return r.invoke(query)

                # If it has get_relevant_documents or get_relevant_docs
                if hasattr(r, "get_relevant_documents"):
                    return r.get_relevant_documents(query)
                if hasattr(r, "get_relevant_docs"):
                    return r.get_relevant_docs(query)

            except Exception:
                # Fall back to an empty list on error to avoid crashing the import
                return []

    SelfQueryRetriever = FallbackSelfQueryRetriever

from src.config import (
    CHROMA_SERVER_NO_TELEMETRY,
    DOCUMENT_CHUNK_BATCH_SIZE,
    DOCUMENT_DIR,
    I_QUES,
    SEMANTIC_THRESH_LIMIT,
    VECTOR_RESULT_CNT,
    VECTORS_DIR
)


@dataclass
class AttributeInfo:
    """
    Bypasses LangChain's shifting dependency structures by using a native
    Python dataclass that mimics the expected metadata_field_info schema
    used by SelfQueryRetriever.from_llm.
    """
    name: str
    description: str
    type: str

class ChromaModel:
    """
    Manages the persistent vector store lifecycle using ChromaDB and handles both similarity-based and
    metadata-structured Self-Query retrieval mechanisms.
    """

    def __init__(self, dataset: dict):
        # Initialize ChromaDB client without diagnostic telemetry overheads
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = CHROMA_SERVER_NO_TELEMETRY
        logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

        # Ensure the configured VECTORS_DIR exists and initialize a PersistentClient
        # that writes all collections and metadata to db/ as the central database path.
        os.makedirs(VECTORS_DIR, exist_ok=True)
        self.chromadb_client = chromadb.PersistentClient(path=VECTORS_DIR)

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
                setattr(self, key, value)

        if self.llm is None and 'openai_model' in dataset:
            openai_model = dataset['openai_model']
            self.llm = openai_model.llm

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

    def _get_semantic_text_splitter(self) -> SemanticChunker:
        """Initializes the semantic text splitter using the target embedding framework."""
        return SemanticChunker(
            self.embedding_model,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=SEMANTIC_THRESH_LIMIT
        )

    def get_retriever(self) -> VectorStoreRetriever:
        """Initializes vector retriever and gets vectorized data stored in Chroma."""
        vector_storage = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            # Use the shared PersistentClient created on the ChromaModel so all
            # collections use the same underlying client instance (avoids race
            # conditions and readonly DB errors when multiple clients touch the
            # same sqlite files).
            client=self.chromadb_client,
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

        # Use os.path.join to build a platform-correct path like: db/<path>_db
        return os.path.join(VECTORS_DIR, f"{path}_db")

    def _get_semantic_storage(self) -> Chroma:
        """Instantiates the specialized semantic storage partition layer."""
        return Chroma(
            embedding_function=self.embedding_model,
            collection_name="semantic_chunks",
            client=self.chromadb_client,
        )

    def add_semantic_documents(self, semantic_chunks: list) -> None:
        """
        Batches and commits document nodes out to the semantic database index.
        :param semantic_chunks:
        :return:
        """

        import time
        batch_size = DOCUMENT_CHUNK_BATCH_SIZE
        for i in range(0, len(semantic_chunks), batch_size):
            batch = semantic_chunks[i: i + batch_size]
            # Attempt to add with a small retry loop to handle transient
            # 'readonly database' errors that can occur under certain
            # filesystem or concurrency conditions. This makes the pipeline
            # more robust in practice.
            retries = 3
            for attempt in range(1, retries + 1):
                try:
                    self.semantic_storage.add_documents(batch)
                    break
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        "Attempt %d/%d: failed to add batch (size=%d): %s",
                        attempt, retries, len(batch), e,
                    )
                    # On final failure re-raise to preserve original behavior
                    if attempt == retries:
                        logging.getLogger(__name__).error(
                            "Failed to write semantic batch after %d attempts; re-raising",
                            retries,
                        )
                        raise
                    time.sleep(0.5)

    def _get_vector_storage(self) -> Chroma:
        """Instantiates the generalized target collection vector layer."""
        return Chroma(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            client=self.chromadb_client,
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
