# src/doc_metadata.py
from datetime import datetime, UTC
import hashlib
import json
import os
import platform
import uuid
from src.constants import DOCUMENT_FILEPATH, LLAMA_MODEL
from src.utils import gen_uuid

class DocumentMetadata:

    """
    Handles metadata for semantic and vector documents
    """

    def __init__(self, dataset: dict, page_content: str):
        # 1. Initialize properties using internal backing variables
        self._doc_id = gen_uuid()
        self._page = 0
        self._page_label = ""
        self._source = ""
        self._total_pages = 0
        self._type = "Document"
        self._utc_datetime = str(datetime.now(UTC))
        self.chunk_id = ""
        self.chunk_seq = 0
        self.creator = ""
        self.doc_type = ""
        self.document_domain = "medical"
        self.generation_model = LLAMA_MODEL
        self.moddate = ""
        self.original_page_content = page_content
        self.parent_id = ""
        self.producer = ""
        self.version = "19th Edition"

        # Map incoming data cleanly
        self._set_attrs(dataset)

    def _set_attrs(self, dataset: dict) -> None:
        """Safely maps dataset keys to class attributes while preventing crashes on

        read-only properties.
        """
        excluded_keys = {
            "checksum",
            "creationdate",
            "doc_id",
            "filename",
            "utc_datetime",
        }

        for key, value in dataset.items():
            if key in excluded_keys:
                continue

            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)

    @property
    def doc_id(self) -> str:
        return self._doc_id

    @property
    def utc_datetime(self) -> str:
        return self._utc_datetime

    @property
    def page(self) -> int:
        try:
            return int(self._page)
        except (ValueError, TypeError):
            return 0

    @page.setter
    def page(self, value):
        self._page = value

    @property
    def page_label(self) -> str:
        if not self._page_label or str(self._page_label).strip() == "":
            return str(self.page + 1)
        return str(self._page_label).strip()

    @page_label.setter
    def page_label(self, value):
        self._page_label = value

    @property
    def total_pages(self) -> int:
        try:
            return int(self._total_pages)
        except (AttributeError, ValueError, TypeError):
            return 0

    @total_pages.setter
    def total_pages(self, value):
        self._total_pages = value

    @property
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, value):
        self._source = value

    @property
    def filename(self) -> str:
        if self._source:
            return os.path.basename(self._source)
        return ""

    @property
    def text_id(self):
        if self.doc_type == 'semantic_chunk':
            clean_fn = self.filename.replace(".", "_")
            seq = getattr(self, 'chunk_seq', 0)
            return f"parent__{clean_fn}__p{self.page}__c{seq}__text"
        return ""

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def creationdate(self) -> str:
        try:
            file_info = os.stat(DOCUMENT_FILEPATH)
            if platform.system() == "Windows":
                timestamp = file_info.st_ctime
            else:
                timestamp = getattr(
                    file_info, "st_birthtime", file_info.st_mtime
                )
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        except Exception:
            return ""

    @property
    def checksum(self) -> str:
        """Generates a stable data hash, safely selecting the correct text payload

        based on document types to prevent data indexing collisions.
        """
        excluded_keys = {"_doc_id", "doc_id", "id", "chunk_id", "text_id"}

        # Strips out internal underscores correctly to stabilize JSON serialization keys
        filtered_metadata = {
            key.lstrip("_"): value
            for key, value in self.__dict__.items()
            if key not in excluded_keys
               and key.lstrip("_") not in excluded_keys
               and "date" not in key.lower()
        }

        # 🎯 Route the text hashing target based on document hierarchy
        filtered_metadata["original_page_content"] = self.original_page_content.strip() if self.original_page_content else ""

        metadata_json = json.dumps(
            filtered_metadata, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()

    @property
    def to_dict(self) -> dict:
        """
        Serializes the metadata to a dictionary, automatically filtering out empty
        attributes to return context-specific keys based on the doc_type.
        """
        property_fields = [
            "checksum", "creationdate", "doc_id", "doc_type", "filename",
            "page", "page_label", "source", "text_id", "total_pages", "type", "utc_datetime"
        ]
        data = {field: getattr(self, field) for field in property_fields}

        # Merge in public attributes from __dict__
        data.update({k: v for k, v in self.__dict__.items() if not k.startswith("_") and k != 'original_page_content'})

        # Filter out None and empty strings
        cleaned = {k: v for k, v in data.items() if v not in (None, "")}

        # Final sort
        did_val = cleaned.pop("doc_id", self._doc_id)
        return {"doc_id": did_val, **dict(sorted(cleaned.items()))}
