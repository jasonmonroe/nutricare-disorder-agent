# src/doc_handler.py

import hashlib
import json
import os
import random
import re
import glob

# Vendor Libraries
# LangChain Imports
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_core.documents import Document  # Document data structures
from llama_parse import LlamaParse  # Document parsing library

# Local Libraries
from src.config import DOCUMENT_DIR, VENDORS_DIR


class DocHandler():
    def __init__(self, llama_parser: LlamaParse):
        self.documents = [] # Set outside the class
        self.document_content_description = "Text Semantic Chunks for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization"
        self.folder_path = DOCUMENT_DIR
        self.metadata_info = self._get_metadata_info()

        self.__wipe_db_dir()

        json_objs = self._parse(llama_parser)
        self.page_texts, self.tables = self._extract_tables(json_objs)
        
    def create(self, content: str, metadata: dict) -> Document:
        """
        Creates and returns a Document object with metadata
        see: https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html

        :param content:
        :param metadata:
        :return:
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

    # Generate a unique ID for document content
    def gen_id(self, source: str, page_no: int, content: str) -> str:
        id_str = f"{source}|page{page_no}|{content[:64]}"

        return str(hashlib.sha256(id_str.encode('utf-8')).hexdigest())

    # Show a random document sample
    def show_sample(self, samp_docs: list, samp_title: str = "") -> None:
        doc_cnt = len(samp_docs)
        index = random.randint(0, doc_cnt - 1)
        print(f"Index = {index}, Count = {doc_cnt}")

        # Check if the index is within bounds
        if 0 <= index < doc_cnt:
            print("ID: ", samp_docs[index].id, "\n")
            print("Metadata:")
            print(json.dumps(samp_docs[index].metadata, indent=4), "\n")
            print(f"{samp_title}:\n", samp_docs[index].page_content)

        else:
            print(f"\nIndex {index} is out of range for the list with length {doc_cnt}.")

    def get_semantic_chunks(self, semantic_chunks):
        return [self.create(d.page_content, d.metadata) for i, d in enumerate(semantic_chunks)]

    #
    def show_documents(self) -> None:
        # Display retrieved documents
        for i in self.documents:
            print("Source:", i.metadata['source'])   # Fill in the correct key for source (e.g., 'source')
            print("Page:", i.metadata['page'], "\n") # Fill in the correct key for page number (e.g., 'page')
            print("Page Content:", i.page_content)
            print("---\n")

    def _parse(self, llama_parser: LlamaParse) -> list:
        # Parse content from PDFs
        # List to store parsed JSON objects
        json_objs = []

        # Define the folder containing the documents

        # Iterate through PDFs in the folder and parse content
        for pdf in os.listdir(self.folder_path):
            if pdf.endswith(".pdf"):
                pdf_path = os.path.join(self.folder_path, pdf)
                json_objs.extend(llama_parser.get_json_result(pdf_path))

        # Show objs
        print(json.dumps(json_objs[0], indent=4))

        return json_objs

    def _extract_tables(self, json_objs) -> tuple:
        """
        Revised Cell to properly get text from document pages: {page_texts}.
        Initialize dictionaries to store page texts and tables.

        :param json_objs:
        :return:
        """

        page_texts, tables = {}, {}

        # Extract tables and adjacent text from the parsed JSON objects
        for obj in json_objs:
            json_list = obj['pages']
            name = obj["file_path"].split("/")[-1]  # Extract the file name

            page_texts[name] = {}
            tables[name] = {}

            for json_item in json_list:
                page_number = json_item["page"]

                # 1. Check for and store table data from the 'items' array
                table_ref_string = None
                table_rows = None

                for component in json_item['items']:
                    if component['type'] == 'table':  # Check if the component is a table
                        table_rows = component['rows']
                        # The text field in the item component often holds the table title/reference
                        # This is the string we need to remove from the overall page text.
                        # We'll rely on the main page 'text' field for removal.

                        # Check the overall page text for a table reference string
                        # We will search for a pattern like '[Table 1-1. ...]'

                        # This pattern finds any text inside brackets that looks like a table reference
                        # We look at the 'text' field of the component if it exists, otherwise rely on manual inspection.

                        # This pattern searches the page text for the exact table title
                        try:
                            # Find the table title text just before the table component starts
                            # We can use the text from the previous item or just the general table placeholder text if available
                            # Based on your example, the text is '[Table 1-1. Glycemic Index of Some Foods]'
                            table_ref_string = next(
                                (
                                    item['value'] for item in json_item['items']
                                    if item['type'] == 'text' and 'Table' in item['value']
                                ), None
                            )

                            # A more reliable way based on the overall page text:
                            table_ref_match = re.search(r'(\[Table\s*[\d\-\.]+\.\s*[^\]]+\])', json_item['text'])
                            if table_ref_match:
                                table_ref_string = table_ref_match.group(1)

                        except Exception as e:
                            print(f"No table ref string. Error: {e}")
                            table_ref_string = None  # Failed to find the specific table reference text

                        # Store the table data
                        tables[name][page_number] = table_rows
                        break  # Assuming max one table per page for context extraction

                # 2. Extract the clean adjacent text context
                page_content_full = json_item['text']
                page_content_clean = page_content_full

                # Remove the table reference string if found
                if table_rows and table_ref_string:
                    page_content_clean = page_content_full.replace(table_ref_string, "").strip()

                # If no explicit reference string was found, we still need to strip the content
                # For simplicity, if a table exists, we remove common boilerplate text around tables
                elif table_rows:
                    # Look for lines that contain the table header/title implicitly
                    lines = page_content_full.split('\n')
                    clean_lines = []

                    for line in lines:
                        # Heuristically remove lines that look like table headers or footers
                        if 'Table' in line and any(c.isdigit() for c in line):
                            continue

                        # Also remove the page numbers/footers
                        if line.strip().isdigit() and len(line.strip()) < 4:
                            continue

                        clean_lines.append(line)

                    page_content_clean = "\n".join(clean_lines).strip()

                # Store the final context text
                page_texts[name][page_number] = page_content_clean

        print(json.dumps(page_texts, indent=4))

        return page_texts, tables

    def show_tables(self) -> None:
        tables = self.tables
        # Display extracted tables for each PDF
        for file_name, file_tables in tables.items():
            print(f"Tables from {file_name}:")
            for page_num, table_rows in file_tables.items():
                print(f"Page {page_num}:")
                for row in table_rows:
                    print(f"\t{row}")

    def _get_metadata_info(self) -> list:
        return [
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
        ]

    def __wipe_db_dir():
        # Wipe all data in the db directory so that we will have a clean slate.
        files = glob.glob(VENDORS_DIR)
        for f in files:
            os.remove(f)

