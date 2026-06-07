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
import glob
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
    DOCUMENT_ZIP,
    VECTORS_DIR, 
    I_DB, 
    I_DISK,
    I_CROSSMARK, 
    I_DOCUMENT, 
    I_FLAG, 
    I_CHECKMARK
)


class DocHandler():
    def __init__(self, llama_parser: LlamaParse):
        self.documents = []  # Explicit state tracking placeholder (Set outside the class)
        self.document_content_description = "Text Semantic Chunks for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization"
        self.folder_path = DOCUMENT_DIR
        self.metadata_info = self._get_metadata_info()
        self.__wipe_db_dir()

        if self._unzip():
            json_objs = self._parse(llama_parser)
            self.page_texts, self.tables = self._extract_tables(json_objs)
        
    def create(self, content: str, metadata: dict) -> Document:
        """
        Creates and returns a LangChain Document object packed with metadata.
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
        FIXED: Removed broken 'self' positional tracking reference parameter.
        """
        id_str = f"{source}|page{page_no}|{content[:64]}"
        return str(hashlib.sha256(id_str.encode('utf-8')).hexdigest())

    def show_sample(self, samp_docs: list, samp_title: str = "") -> None:
        """Displays a random processed document entity validation footprint."""
        doc_cnt = len(samp_docs)
        if doc_cnt == 0:
            print("[WARNING] Checked baseline collection is empty.")
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
        """Maps continuous semantic raw chunks into formal wrapped LangChain Documents."""
        return [self.create(d.page_content, d.metadata) for d in semantic_chunks]

    def show_documents(self) -> None:
        """Utility visualization logger looping structural collection layers."""
        for i in self.documents:
            print("Source:", i.metadata.get('source', 'Unknown'))
            print("Page:", i.metadata.get('page', 'Unknown'), "\n")
            print("Page Content:", i.page_content)
            print("---\n")

    def _parse(self, llama_parser: LlamaParse) -> list:
        """Loads physical dataset storage and feeds buffers through LlamaParse pipelines."""
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
            print(f"{I_CHECKMARK} [SUCCESS] Sample file parsed metadata header structured.")
        else:
            print(f"{I_FLAG} No document payload arrays fetched.")

        return json_objs

    def _extract_tables(self, json_objs: list) -> tuple[dict, dict]:
        """
        Orchestrates extracting tables and processing corresponding adjacent clear strings.
        Successfully refactored implementation logic branches.
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

                # 1. Isolate embedded matrix layers and structural title labels
                table_rows, table_ref_string = self._process_page_tables(json_item)

                if table_rows:
                    tables[name][page_number] = table_rows

                # 2. Re-route pure extraction strings into cleanup components
                page_content_full = json_item.get('text', '')
                cleaned_text = self._clean_page_text(page_content_full, table_rows, table_ref_string)

                page_texts[name][page_number] = cleaned_text

        return page_texts, tables

    def _process_page_tables(self, json_item: dict) -> tuple[list | None, str | None]:
        """
        Refactored page context extractor checking table markers and resolving title strings.
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

    def _clean_page_text(self, page_content_full: str, table_rows: list | None, table_ref_string: str | None) -> str:
        """
        Pure algorithmic string cleaning component transforming texts based on page metadata structure.
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
        for file_name, file_tables in self.tables.items():
            print(f"Tables from {file_name}:")
            for page_num, table_rows in file_tables.items():
                print(f"Page {page_num}:")
                for row in table_rows:
                    print(f"\t{row}")

    @staticmethod
    def _get_metadata_info() -> list[AttributeInfo]:
        """Provides metadata typing validation mappings configuration layers."""
        return [
            AttributeInfo(name="page", description="page number of document", type="integer"),
            AttributeInfo(name="source", description="file path of document", type="string"),
            AttributeInfo(name="page_content", description="raw text of (sectional) document", type="string")
        ]

    @staticmethod
    def __wipe_db_dir() -> None:
        """
        Wipes data inside target persistent storage directories to allow clean ingestion.
        FIXED: Uses recursive shutil tree removal to clear nested ChromaDB states safely.
        """
        print(f"{I_DB} # --- Wiping {VECTORS_DIR} --- # {I_DB}")

        # Target internal database children dynamically to maintain database directories cleanly
        if os.path.exists(VECTORS_DIR):
            for filename in os.listdir(VECTORS_DIR):
                file_path = os.path.join(VECTORS_DIR, filename)
                try:
                    print(f"{I_CROSSMARK} Wiping {I_DOCUMENT} internal component: {file_path} ...")
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"{I_FLAG} [ERROR] Failed to wipe element path target {file_path}. Exception: {e}")


    def unzip():
        print(f'\nUnzipping {I_DISK} {DOCUMENT_ZIP}...')
    
        # Unzipping the nutrition medical reference documents into the Nutritional Medical Reference folder
        # Loading the temp.zip and creating a zip object
        with ZipFile(DOCUMENT_ZIP, 'r') as zip_handle:
            # Extracting specific file in the zip into a specific location.
            zip_handle.extract(
                DOCUMENT_FILE,
                path=DOCUMENT_DIR
            )
            zip_handle.close()
            
            sleep(1)

        # Check if file successfully unzipped
        if os.path.exist(str(DOCUMENT_DIR + '/' + DOCUMENT_FILE)):
            print(f'{I_DOCUMENT}{DOCUMENT_DIR}/{DOCUMENT_FILE} successfully unzipped and ready for processing.')
            return True

        print(f'{I_FLAG}{DOCUMENT_FILE} not unzipped!')
        return False
