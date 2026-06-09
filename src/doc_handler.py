# src/doc_handler.py

# +----------------------+
# |     DATA HANDLER     |
# +----------------------+

# Python Libraries
import hashlib
import json
import os
import random
import re
import shutil
from time import sleep
from zipfile import ZipFile

# Vendor Libraries
# LangChain Imports
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_core.documents import Document  # Document data structures
from llama_parse import LlamaParse  # Document parsing library

# Local Libraries
from src.config import (
    DOCUMENT_DIR, 
    DOCUMENT_FILE, 
    DOCUMENT_FILEPATH,
    DOCUMENT_ZIP,
    I_BROOM,
    I_CHECKMARK,
    I_DB,
    I_DISK,
    I_DOCUMENT,
    I_FLAG,
    I_WARNING,
    VECTORS_DIR,
)


class DocHandler():
    def __init__(self, llama_parser: LlamaParse):
        self.documents = []
        self.document_content_description = "Text Semantic Chunks for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization"
        self.folder_path = DOCUMENT_DIR
        self.metadata_info = self._get_metadata_info()
        
        if self._unzip():
            json_objs = self._parse(llama_parser)
            self.page_texts, self.tables = self._extract_tables(json_objs)
        
    def create(self, content: str, metadata: dict) -> Document:
        """
        Creates and returns a LangChain Document object packed with metadata.
        :param content:
        :param metadata:
        :return: Document
        """

        metadata["doc_id"] = self.gen_id(metadata["source"], metadata["page"], content)

        if "type" not in metadata:
            metadata["type"] = "Document"

        return Document(
            id=metadata["doc_id"],
            page_content=content,
            metadata=metadata,
            type=metadata["type"],
        )

    @staticmethod
    def gen_id(source: str, page_no: int, content: str) -> str:
        """
        Generates a deterministic unique SHA-256 fingerprint ID for a semantic chunk.

        :param source:
        :param page_no:
        :param content:
        :return:
        """

        id_str = f"source:{source}|page{page_no}|content:{content[:64]}"
        return str(hashlib.sha256(id_str.encode('utf-8')).hexdigest())

    def show_sample(self, samp_docs: list, samp_title: str = "") -> None:
        """
        Displays a random processed document entity validation footprint.

        :param samp_docs:
        :param samp_title:
        :return: None
        """

        print(f'\n# --- {I_DOCUMENT} Show sample documents {I_DOCUMENT} --- #')
        
        doc_cnt = len(samp_docs)
        if doc_cnt == 0:
            print(f"{I_WARNING} Checked baseline collection is empty.")
            return

        index = random.randint(0, doc_cnt - 1)
        print(f"Index = {index}, Count = {doc_cnt}")

        if 0 <= index < doc_cnt:
            print("ID: ", samp_docs[index].id, "\n")
            print("Metadata:")
            print(json.dumps(samp_docs[index].metadata, indent=4), "\n")
            print(f"{samp_title}:\n", samp_docs[index].page_content)
        else:
            print(f"\nIndex {index} is out of range for the list with length {doc_cnt}.")

    def get_semantic_chunks(self, semantic_chunks: list) -> list[Document]:
        """
        Maps continuous semantic raw chunks into formal wrapped LangChain Documents.

        :param semantic_chunks:
        :return: Documents
        """

        return [self.create(d.page_content, d.metadata) for d in semantic_chunks]

    def show_documents(self) -> None:
        """Utility visualization logger looping structural collection layers."""
        
        print(f'\n# --- {I_DOCUMENT} Showing Documents {I_DOCUMENT} --- #')
        for i in self.documents:
            print("Source:", i.metadata.get('source', 'Unknown'))
            print("Page:", i.metadata.get('page', 'Unknown'), "\n")
            print("Page Content:", i.page_content)
            print("---\n")

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

        for pdf in os.listdir(self.folder_path):
            if pdf.endswith(".pdf"):
                pdf_path = os.path.join(self.folder_path, pdf)
                print(f"{I_DOCUMENT} Parsing file target: {pdf_path}")
                json_objs.extend(llama_parser.get_json_result(pdf_path))

        if json_objs:
            print(f"{I_CHECKMARK} Sample file parsed metadata header structured.")
        else:
            print(f"{I_FLAG} No document payload arrays fetched.")

        return json_objs

    def _extract_tables(self, json_objs: list) -> tuple[dict, dict]:
        """
        Orchestrates extracting tables and processing corresponding adjacent clear strings.
        Successfully refactored implementation logic branches.

        :param json_objs:
        :return: tuple
        """

        page_texts, tables = {}, {}

        for obj in json_objs:
            # Extract file identification name safely
            file_path_str = obj.get("file_path", "unknown_source.pdf")
            name = file_path_str.split("/")[-1]

            page_texts[name] = {}
            tables[name] = {}

            for json_item in obj.get('pages', []):
                page_number = json_item.get("page")

                # Isolate embedded matrix layers and structural title labels
                table_rows, table_ref_string = self._process_page_tables(json_item)

                if table_rows:
                    tables[name][page_number] = table_rows

                # Re-route pure extraction strings into cleanup components
                page_content_full = json_item.get('text', '')
                cleaned_text = self._clean_page_text(page_content_full, table_rows, table_ref_string)

                page_texts[name][page_number] = cleaned_text

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
                    flag_symbol = self.I_FLAG if hasattr(self, 'I_FLAG') else '[FLAG]'
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

    def show_tables(self) -> None:
        """Displays formatted representation profiles of isolated layout data tables."""
        
        print(f'\n# --- {I_DB} Showing Table Information {I_DB} --- #')
        for file_name, file_tables in self.tables.items():
            print(f"Tables from {file_name}:")
            for page_num, table_rows in file_tables.items():
                print(f"Page {page_num}:")
                for row in table_rows:
                    print(f"\t{row}")

    @staticmethod
    def _get_metadata_info() -> list[AttributeInfo]:
        """
        Provides metadata typing validation mappings configuration layers.

        :return: list
        """

        return [
            AttributeInfo(name="page", description="page number of document", type="integer"),
            AttributeInfo(name="source", description="file path of document", type="string"),
            AttributeInfo(name="page_content", description="raw text of (sectional) document", type="string")
        ]

    @staticmethod
    def wipe_db_dir() -> None:
        """
        Wipes data inside target persistent storage directories to allow clean ingestion.
        FIXED: Uses recursive shutil tree removal to clear nested ChromaDB states safely.
        """
        print(f"\n# --- {I_BROOM} Wiping {VECTORS_DIR} {I_BROOM} --- #")

        if not os.path.exists(VECTORS_DIR):
            # Create fresh db directory
            print(f'{I_WARNING} {VECTORS_DIR} does not exist.  Creating {I_DB} it now...')
            os.makedirs(VECTORS_DIR, exist_ok=True)
            return None

        # Target internal database children dynamically to maintain database directories cleanly
        if os.path.exists(VECTORS_DIR):
            for filename in os.listdir(VECTORS_DIR):
                file_path = os.path.join(VECTORS_DIR, filename)
                try:
                    print(f"{I_CHECKMARK} Wiping {I_DOCUMENT} internal component: {file_path} ...")
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"{I_FLAG} Failed to wipe element path target {file_path}. Exception: {e}")
           
        if next(os.scandir(VECTORS_DIR), None) is None:
            print("Directory exists and is empty.")

        return None

    @staticmethod
    def _unzip() -> bool:
        """Extracts reference files dynamically from the project zip archive.
        
        Uses DOCUMENT_FILEPATH for local path verification, while inspecting the 
        internal zip manifest to safely handle folder-nested contents inside the archive.
        """
        # If the file already exists at our new explicit path, skip extraction entirely
        if not os.path.exists(DOCUMENT_FILEPATH):
            print(f'\nUnzipping {I_DISK} {DOCUMENT_ZIP}...')
            
            # Double check that the zip file actually exists before trying to read it
            if not os.path.exists(DOCUMENT_ZIP):
                print(f'{I_FLAG} Source archive file {DOCUMENT_ZIP} does not exist!')
                return False

            # FIXED: Using ZipFile directly to match your top-level import
            with ZipFile(DOCUMENT_ZIP, 'r') as zip_handle:
                # 1. Search the zip manifest array for any entry ending with our filename
                archive_target_key = None
                for member in zip_handle.namelist():
                    if member.endswith(DOCUMENT_FILE):
                        archive_target_key = member
                        break

                # If the file wasn't found anywhere inside the archive, exit gracefully
                if not archive_target_key:
                    print(f'{I_FLAG} Could not find {DOCUMENT_FILE} anywhere inside the archive!')
                    return False

                # 2. Ensure destination folder exists, read the zip stream, and write 
                # it out directly to your new clean local path constant
                os.makedirs(DOCUMENT_DIR, exist_ok=True)
                with zip_handle.open(archive_target_key) as source_stream:
                    with open(DOCUMENT_FILEPATH, 'wb') as dest_file:
                        dest_file.write(source_stream.read())
                
                sleep(1)

        # Final logic boundary check utilizing your new path constant
        if os.path.exists(DOCUMENT_FILEPATH):
            print(f'{I_DOCUMENT} {DOCUMENT_FILEPATH} successfully unzipped and ready for processing.')
            return True

        print(f'{I_FLAG} {DOCUMENT_FILE} not unzipped!')
        return False
