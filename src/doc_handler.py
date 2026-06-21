from __future__ import annotations
# src/doc_handler.py

# +----------------------+
# |     DATA HANDLER     |
# +----------------------+

# Python Libraries
import hashlib
import json
import os
import platform
import random
import re
import shutil
import uuid
from datetime import datetime, UTC
from time import sleep
from zipfile import ZipFile

# Vendor Libraries
# LangChain Imports
from langchain_core.documents import Document  # Document data structures
from llama_parse import LlamaParse  # Document parsing library

# Local Libraries
from src.constants import (
    DOCUMENT_DIR,
    DOCUMENT_DIR_PERM, 
    DOCUMENT_FILE, 
    DOCUMENT_FILEPATH,
    DOCUMENT_ZIP,
    I_BOOK,
    I_BROOM,
    I_CHECKMARK,
    I_DB,
    I_DIR,
    I_DISK,
    I_DOCUMENT,
    I_FLAG,
    I_PEN,
    I_WARNING
)
from src.doc_metadata import DocumentMetadata

from src.model_config import config
from src.utils import show_timer, start_timer, gen_uuid


class DocHandler():
    def __init__(self, llama_parser: LlamaParse, skip_parse: bool = False):
        self.document_chunks = []
        self.documents = [] # similarity searched documents
        self.folder_path = DOCUMENT_DIR
        self.creation_date = self._get_creation_date()
        self.page_texts, self.tables = {}, {}

        if not skip_parse and self._unzip():
            json_objs = self._parse(llama_parser)
            self.page_texts, self.tables = self._extract_tables(json_objs)

    def _parse(self, llama_parser: LlamaParse) -> list:
        """
        Loads physical dataset storage and feeds buffers through LlamaParse pipelines.

        :param llama_parser:
        :return: list
        """

        json_objs = []
        if not os.path.exists(self.folder_path):
            print(f"{I_FLAG} Data folder path location '{self.folder_path}' doesn't exist.")
            return json_objs

        start_time = start_timer()
        for pdf in os.listdir(self.folder_path):
            if pdf.endswith(".pdf"):
                pdf_path = os.path.join(self.folder_path, pdf)
                print(f"{I_DOCUMENT} Parsing file target: {pdf_path}")
                json_objs.extend(llama_parser.get_json_result(pdf_path))

        if json_objs:
            print(f"{I_CHECKMARK} Sample file parsed metadata header structured.")
        else:
            print(f"{I_FLAG} No document payload arrays fetched.")

        show_timer(start_time)

        return json_objs

    def _extract_tables(self, json_objs: list) -> tuple[dict, dict]:
        """Orchestrates extracting tables and processing corresponding adjacent clear strings.

        Successfully refactored implementation logic branches with explicit tracking IDs.
        """
        page_texts, tables = {}, {}

        for obj in json_objs:
            file_path_str = obj.get("file_path", "unknown-source.pdf")
            clean_filename = os.path.basename(file_path_str).replace(".", "_")

            # Initialize file path tracking roots
            page_texts[file_path_str] = {}
            tables[file_path_str] = {}

            for json_item in obj.get('pages', []):
                page_number = json_item.get("page")

                # Isolate embedded matrix layers
                table_rows, table_ref_string = self._process_page_tables(json_item)

                # Build predictable, identifiable IDs instead of random UUIDs
                # This makes downstream relational child lookup trivial!
                text_id = f"parent__{clean_filename}__p{page_number}__text"
                table_id = f"parent__{clean_filename}__p{page_number}__table"

                # Store text with its explicit ID package
                page_content_full = json_item.get('text', '')
                cleaned_text = self._clean_page_text(page_content_full, table_rows, table_ref_string)

                page_texts[file_path_str][page_number] = {
                    "text_id": text_id,
                    "content": cleaned_text
                }

                # Store table with its explicit ID package (only if table data exists)
                if table_rows:
                    tables[file_path_str][page_number] = {
                        "table_id": table_id,
                        "rows": table_rows
                    }

        return page_texts, tables

    # @todo - defunct
    def _extract_tables_orig(self, json_objs: list) -> tuple[dict, dict]:
        """
        Orchestrates extracting tables and processing corresponding adjacent clear strings.
        Successfully refactored implementation logic branches.

        :param json_objs:
        :return: tuple
        """

        page_texts, tables = {}, {}
        for obj in json_objs:
            # ✅ Extract full file path string safely from the source JSON object
            file_path_str = obj.get("file_path", "unknown-source.pdf")

            # Use the full path as the primary key dictionary anchor
            page_texts[file_path_str] = {}
            tables[file_path_str] = {}
            # Add a unique identifier @todo -look into this
            tables['table_id'] = gen_uuid()

            for json_item in obj.get('pages', []):
                page_number = json_item.get("page")

                # Isolate embedded matrix layers and structural title labels
                table_rows, table_ref_string = self._process_page_tables(json_item)

                if table_rows:
                    tables[file_path_str][page_number] = table_rows

                # Re-route pure extraction strings into cleanup components
                page_content_full = json_item.get('text', '')
                cleaned_text = self._clean_page_text(page_content_full, table_rows, table_ref_string)

                page_texts[file_path_str][page_number] = cleaned_text

                # Add a unique identifier @todo -look into this
                page_texts[file_path_str]['text_id'] = gen_uuid()

        return page_texts, tables

    def _process_page_tables(self, json_item: dict) -> tuple[list | None, str | None]:
        """
        Refactored page context extractor checking table markers and resolving title strings.
        :param json_item:
        :return: tuple
        """

        table_rows = None
        table_ref_string = None

        has_table = any(comp.get('type') == 'table' for comp in json_item.get('items', []))
        if not has_table:
            return None, None

        for component in json_item.get('items', []):
            if component.get('type') == 'table':
                table_rows = component.get('rows')

                try:
                    # Target layout components for matching table title descriptions
                    table_ref_string = next(
                        (
                            item['value'] for item in json_item.get('items', [])
                            if item.get('type') == 'text' and 'Table' in str(item.get('value', ''))
                        ), None
                    )

                    # Regex match guard step targeting literal raw content brackets
                    table_ref_match = re.search(r'(\[Table\s*[\d\-\.]+\.\s*[^\]]+\])', json_item.get('text', ''))
                    if table_ref_match:
                        table_ref_string = table_ref_match.group(1)

                except Exception as e:
                    # Graceful exception logging boundary handling
                    flag_symbol = I_FLAG
                    print(f"{flag_symbol} No table ref string. Error: {e}")
                    table_ref_string = None

                break  # Enforce processing limit threshold context trace constraint

        return table_rows, table_ref_string

    @staticmethod
    def _clean_page_text(page_content_full: str, table_rows: list | None, table_ref_string: str | None) -> str:
        """
        Pure algorithmic string cleaning component transforming texts based on page metadata structure.

        :param page_content_full:
        :param table_rows:
        :param table_ref_string:
        :return:
        """

        # Scenario A: Structured key elements confirmed, replace explicit target reference
        if table_rows and table_ref_string:
            return page_content_full.replace(table_ref_string, "").strip()

        # Scenario B: Table detected, run line heuristic filters to drop headers and margins
        if table_rows:
            lines = page_content_full.split('\n')
            clean_lines = []

            for line in lines:
                if 'Table' in line and any(c.isdigit() for c in line):
                    continue
                if line.strip().isdigit() and len(line.strip()) < 4:
                    continue
                clean_lines.append(line)

            return "\n".join(clean_lines).strip()

        # Scenario C: Stable ordinary string sheet, return flat asset normalized
        return page_content_full.strip()

    def count_document_chunks(self) -> int:
        return len(self.document_chunks)

    def get_semantic_chunks(self, semantic_chunks: list) -> list[Document]:
        """
        Maps continuous semantic raw chunks into formal wrapped LangChain Documents.
        Tracks a per-page sequence counter to guarantee unique text_ids when a single
        page is split into multiple semantic chunks.
        """
        page_chunk_counters = {}  # tracks how many chunks seen per page so far
        documents = []

        for d in semantic_chunks:
            page = d.metadata.get('page', 0)
            seq = page_chunk_counters.get(page, 0)
            page_chunk_counters[page] = seq + 1

            metadata = d.metadata.copy()
            metadata['chunk_seq'] = seq  # 👑 NEW: per-page sequence index

            documents.append(self.create(d.page_content, metadata))

        return documents

    def show_documents(self) -> None:
        """Utility visualization logger looping structural collection layers."""

        print(f'\n# --- {I_BOOK} Showing {len(self.documents)} Documents {I_BOOK} --- #')

        for i, doc in enumerate(self.documents):
            print(f'\n\t{I_DOCUMENT} Document: {i+1} -----')
            print("\t\tSource:", doc.metadata.get('source', 'Unknown'))
            print("\t\tFilename:", doc.metadata.get('filename', 'Unknown filename'))
            print("\t\tPage:", doc.metadata.get('page', 'Unknown'))
            print("\t\tPage Content:", doc.page_content)
            print(f'\t+-- Document: {i+1} ----+')

    def show_tables(self) -> None:
        """Displays formatted representation profiles of isolated layout data tables."""

        print(f'\n# --- {I_DB} Showing Table Information {I_DB} --- #')
        for file_name, file_tables in self.tables.items():
            print(f"\tTables from {file_name}:")
            for page_num, table_rows in file_tables.items():
                print(f"\tPage {page_num}:")
                for row in table_rows:
                    print(f"\t{row}")

    def show_sample(self, samp_docs: list, samp_title: str = "") -> None:
        """
        Displays a random processed document entity validation footprint.

        :param samp_docs:
        :param samp_title:
        :return: None
        """

        print(f'\n# --- {I_DOCUMENT} Show (random) sample {samp_title} document {I_DOCUMENT} --- #')

        doc_cnt = len(samp_docs)
        if doc_cnt == 0:
            print(f"\t{I_WARNING} Checked baseline collection is empty. No sample to show.")
            return

        index = random.randint(0, doc_cnt - 1)
        print(f"\tCount = {doc_cnt}, Index = {index}")

        if 0 <= index < doc_cnt:
            print("\tID: ", samp_docs[index].id, "\n")
            print("\tMetadata:")
            print(json.dumps(samp_docs[index].metadata, indent=4), "\n")
            print(f"\t{samp_title}:\n", samp_docs[index].page_content)
        else:
            print(f"\n\tIndex {index} is out of range for the list with length {doc_cnt}.")



    def create(self, page_content: str, metadata_dataset: dict, flag: bool=False) -> Document:
        """
        Creates the actual document.  We have two document types semantic and vector documents.
        Semantic documents:
        - doc_type: 'semantic_chunk'
        - type: 'Document'
        - filename: 'nutritional-disorders.pdf'
        - text_id:
        - checksum:
        - creationdate: date of file creation
        - utc_datetime: when document was executed
        - doc_id: uuid

        Vector documents:
        - doc_type: hypothetical_questions or table_hypothetical_questions.
        - parent_id:
        - chunk_id:
        - checksum

        :param page_content:
        :param metadata:
        :return:
        """
        # Add additional metadata to create the offical document.

        #print(f'flag = {flag}')

        #if flag:
         #   print("Flagged for child creation")
          #  print(f'init_metadata={metadata_dataset}')

        if isinstance(page_content, dict):
            page_content = json.dumps(page_content)

        metadata = metadata_dataset.copy()

        if "doc_type" not in metadata:
            metadata["doc_type"] = "semantic_chunk"

        metadata_instance = DocumentMetadata(metadata, page_content)
        metadata_dict = metadata_instance.to_dict

        #f flag:
        print(f'\nmetadata_dict={metadata_dict}')
           # print(f'page_content={page_content}')
            #import sys
            #print('Running questions\nTERMINATE!')
            #sys.exit(1)


        #print('END!!!')
        #import sys
        #sys.exit(1)

        return Document(
            id=metadata_dict["doc_id"],
            type=metadata_dict["type"],
            page_content=page_content,
            metadata=metadata_dict,
        )

    # @todo - defunct
    def create_(self, content: str, metadata_dataset: dict) -> Document:
        """Creates a LangChain Document, dynamically adopting its metadata strategy
        based on whether the asset is a raw parent chunk or a synthetic child generation.
        """
        # 👑 DEFENSIVE: Ensure content is a string to prevent Pydantic ValidationErrors

        metadata_dataset = sorted(metadata_dataset)
        print(f'dataset={metadata_dataset}')
        if isinstance(content, dict):
            content = json.dumps(content)

        dataset = metadata_dataset.copy()
        doc_type = dataset.get("doc_type", "semantic_chunk")

        # Strategy A: Use the strict metadata class for core chunks and synthetic question layers
        if doc_type in ["semantic_chunk", "questions", "table_questions"]:
            #print(f'[INFO] Creating Document for type: {doc_type}')

            # For synthetic layers, ensure we have a fresh ID if one wasn't provided
            if doc_type != "semantic_chunk" and "doc_id" not in dataset:
                dataset["doc_id"] = uuid.uuid4().hex.lower()

            metadata_instance = DocumentMetadata(dataset)
            metadata_dict = metadata_instance.to_dict

            # For synthetic layers, we must include content in the checksum
            # to ensure idempotency of LLM generations.
            if doc_type != "semantic_chunk":
                metadata_dict["checksum"] = self._calc_checksum(metadata_dict, content)

        # Strategy B: Use the flexible mapping strategy for child synthetic layers
        else:
            print(f'[DEBUG] Using flexible mapping for doc_type: {doc_type}')
            metadata_dict = {
                "doc_id": dataset.get("doc_id", uuid.uuid4().hex.lower()),
                "doc_type": doc_type,
                "source": dataset.get("source", ""),
                "filename": os.path.basename(dataset.get("source", "")),
                "page": int(dataset.get("page", 0)),
                "parent_id": dataset.get("parent_id", ""),
                "chunk_id": dataset.get("chunk_id", ""),
                "utc_datetime": datetime.now(UTC).isoformat(),
                "type": "Document"
            }
            metadata_dict["checksum"] = self._calc_checksum(metadata_dict, content)

        # Cleanly sort the metadata with doc_id at the top row
        sorted_metadata = {"doc_id": metadata_dict.pop("doc_id"), **dict(sorted(metadata_dict.items()))}

        print(f'sorted_metadata={sorted_metadata}')

        return Document(
            id=sorted_metadata["doc_id"],
            type=metadata_dict["type"],
            page_content=content,
            metadata=metadata_dict,
        )

    # @todo - defunct (child?)??
    def create_semantic(self, metadata:dict):

        # Add additional
        pass

    # @todo - defunct (child?)??
    def create_vector(self):
        """
        metadata_dict = {
            # Existing core tracking keys
            "doc_id": doc_id,
            "doc_type": doc_type, # 'table_hypothetical_questions'
            "source": dataset.get("source", ""),
            "filename": os.path.basename(dataset.get("source", "")),
            "page": int(dataset.get("page", 0)),
            "parent_id": dataset.get("parent_id", ""),
            "chunk_id": dataset.get("chunk_id", ""),
            "utc_datetime": str(datetime.now(UTC)),
            "type": "Document",

            # 🔥 NEW: Search & Sorting Enhancements
            "document_domain": "medical",
            "version": "19th_edition",
            "child_index": int(dataset.get("chunk_id", 0)), # Tracks question index in loop

            # 🔥 NEW: Telemetry & Quality Benchmarking
            "generation_model": "llama-3.1-8b-instant",
            "retrieval_strategy": "parent_jump"
        }
        :return:
        """
        pass

    # @todo - defunct (child?)??
    def create__(self, content: str, metadata_dataset: dict) -> Document:
        """Creates and returns a LangChain Document object packed with metadata.

        see:
        https://reference.langchain.com/python/langchain-core/documents/base/Document
        """
        dataset = metadata_dataset.copy()
        metadata_instance = DocumentMetadata(dataset)
        metadata_dict = metadata_instance.to_dict

        return Document(
            id=metadata_dict["doc_id"],
            type=metadata_dict["type"],
            page_content=content,
            metadata=metadata_dict,
        )

    # @todo - defunct (parent) ??
    def create_semantic_chunks(self, content: str, metadata: dict) -> Document:
        """
        Creates and returns a LangChain Document object packed with metadata.
        see: https://reference.langchain.com/python/langchain-core/documents/base/Document

        :param content:
        :param metadata:
        :return: Document
        """

        metadata = metadata.copy()  # 👑 PROTECT: Prevent in-place mutation of the original dict
        # Add additional keys to the metadata
        if "source" in metadata:
            metadata["filename"] = os.path.basename(metadata["source"])
        elif "filename" in metadata and "source" not in metadata:
            # Fallback to make sure 'source' is never empty if only filename is provided
            metadata["source"] = metadata["filename"]

        # see: https://reference.langchain.com/python/langchain-core/documents/base/Document/type
        if "type" not in metadata:
            metadata["type"] = "Document"

        if "page" in metadata:
            try:
                metadata["page"] = int(metadata["page"])
            except (ValueError, TypeError):
                pass # Fallback if page is not a numeric string

        # Check for duplicates before adding
        metadata["checksum"] = self._calc_checksum(metadata, content)
        metadata["doc_id"] = self._generate_document_id()

        # `creationdate` key originates from PyPDFLoader()
        if not metadata.get("creationdate"):
            metadata["creationdate"] = self.creation_date

        # Get timestamp of chunk
        metadata["utc_datetime"] = str(datetime.now(UTC))

        # Sort metadata but have the doc_id key at the top.
        metadata = {
            "doc_id": metadata.pop("doc_id"),
            **dict(sorted(metadata.items()))
        }

        print(f'metadata={metadata}')
        return Document(
            id=metadata["doc_id"],
            type=metadata["type"],
            page_content=content,
            metadata=metadata,
        )

    # @todo - defunct -
    def _get_creation_date(self) -> str:
        """Retrieves a reliable file creation timestamp across Windows, Mac, and Linux."""
        file_info = os.stat(DOCUMENT_FILEPATH)

        # Handle Windows vs Unix-like OS splits
        if platform.system() == "Windows":
            timestamp = file_info.st_ctime
        else:
            # Mac uses st_birthtime; Linux falls back to st_mtime (modification time)
            timestamp = getattr(file_info, "st_birthtime", file_info.st_mtime)

        # Return clean formatted string
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

    # @todo - defunct -
    def _generate_document_id(self) -> str:
        # Generate UUID without hyphens
        # see: https://reference.langchain.com/python/langchain-core/documents/base/BaseMedia/id
        doc_id = uuid.uuid4().hex

        return doc_id.lower()

    # @todo - defunct ??
    def _calc_checksum(self, metadata: dict, content: str) -> str:
        """Generates a deterministic SHA-256 fingerprint.

        👑 FIX: Excluded 'utc_datetime' from hash calculation to ensure
        the checksum remains stable across different execution runs.
        :param metadata:
        :param content:
        :return:
        """

        # Explicitly isolate changing or non-data keys
        excluded_keys = {"doc_id", "id", "chunk_id", "utc_datetime"}

        filtered_metadata = {
            key: value
            for key, value in metadata.items()
                if key not in excluded_keys and "date" not in key.lower()
        }

        # Add page content to ensure a unique hash for each document
        # Note: content is used for checksum but is not included in metadata intentionally!
        filtered_metadata['page_content'] = content

        # @todo - print(f'filtered_metadata={filtered_metadata}')

        # Serialize to a strictly ordered JSON string to ensure stability
        # separators removes random whitespace differences
        metadata_json = json.dumps(
            filtered_metadata, sort_keys=True, separators=(",", ":")
        )

        # Compute and return the clean SHA-256 string
        return hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()








    @staticmethod
    def wipe_db_dir() -> None:
        """
        Aggressively purges only the target database directory (db/)
        to force a true factory reset of ChromaDB states.
        """

        target_db_dir = os.path.abspath(config.CHROMA_VECTORS_DIR)

        print(f"\n{I_BROOM} Wiping Database Directory: {target_db_dir} {I_BROOM}")

        if not os.path.exists(target_db_dir):
            print(f'{I_WARNING} Directory {target_db_dir} does not exist. \n{I_DIR} Creating a fresh instance now...')
            os.makedirs(target_db_dir, exist_ok=True)
            os.chmod(target_db_dir, DOCUMENT_DIR_PERM)
            return

        # Loop through the children of db/ specifically, leaving data/ completely alone
        for filename in os.listdir(target_db_dir):
            file_path = os.path.join(target_db_dir, filename)
            try:
                print(f"{I_CHECKMARK} Purging database artifact: {file_path} ...")
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"{I_FLAG} Failed to wipe element path target {file_path}. Exception: {e}")

        if next(os.scandir(target_db_dir), None) is None:
            os.chmod(target_db_dir, DOCUMENT_DIR_PERM)
            print(f"{I_DIR} Database directory `{config.CHROMA_VECTORS_DIR}` is completely empty and ready for use!")
            print(f'{I_PEN} {config.CHROMA_VECTORS_DIR} privileges are set to {DOCUMENT_DIR_PERM}.\n')

    @staticmethod
    def _unzip() -> bool:
        """Extracts reference files dynamically from the project zip archive.

        Uses DOCUMENT_FILEPATH for local path verification, while inspecting the
        internal zip manifest to safely handle folder-nested contents inside the archive.
        ℹ️ Note: Full document is 4,114 pages and can be found here:
        https://benhvienanhson.com/media/post_attachments/The_Merck_Manual_of_Diagnosis_and_Therapy_2011_-_19th_Edn........pdf
        """
        # If the file already exists at our new explicit path, skip extraction entirely
        if not os.path.exists(DOCUMENT_FILEPATH):
            print(f'\nUnzipping {I_DISK} {DOCUMENT_ZIP}...')

            # Double check that the zip file actually exists before trying to read it
            if not os.path.exists(DOCUMENT_ZIP):
                print(f'{I_FLAG} Source archive file {DOCUMENT_ZIP} does not exist!')
                return False

            # Using ZipFile directly to match your top-level import
            with ZipFile(DOCUMENT_ZIP, 'r') as zip_handle:
                # Search the zip manifest array for any entry ending with our filename
                archive_target_key = None
                for member in zip_handle.namelist():
                    if member.endswith(DOCUMENT_FILE):
                        archive_target_key = member
                        break

                # If the file wasn't found anywhere inside the archive, exit gracefully
                if not archive_target_key:
                    print(f'{I_FLAG} Could not find {DOCUMENT_FILE} anywhere inside the archive!')
                    return False

                # Ensure destination folder exists, read the zip stream, and write it out directly to your new clean local path constant.
                os.makedirs(DOCUMENT_DIR, exist_ok=True)
                os.chmod(DOCUMENT_DIR, DOCUMENT_DIR_PERM)

                with zip_handle.open(archive_target_key) as source_stream:
                    with open(DOCUMENT_FILEPATH, 'wb') as dest_file:
                        dest_file.write(source_stream.read())

                print(f'\nSleeping for {config.SLEEP_TIME_SEC} seconds...')
                sleep(config.SLEEP_TIME_SEC)

        # --- Check if file was successfully unzipped! --- #
        # Final logic boundary check utilizing your new path constant
        if os.path.exists(DOCUMENT_FILEPATH):
            print(f'{I_DOCUMENT} {DOCUMENT_FILEPATH} successfully unzipped and ready for processing.')
            return True

        print(f'{I_FLAG} {DOCUMENT_FILE} not unzipped!')

        return False





