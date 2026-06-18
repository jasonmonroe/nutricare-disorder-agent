# storages/data_generator.py

from models.chroma import ChromaModel


class DataGenerator:
    def __init__(self, dataset: dict, chroma_db: ChromaModel):
        self.batch_size = 0
        self.chroma_db = chroma_db
        self.collection_name = ''
        self.doc_handle = None
        self.doc_type = ''
        self.llm = None
        self.prompt = ''
        #self.title = ''

        self._set_attrs(dataset)

    def _set_attrs(self, dataset: dict) -> None:
        """
        Safely maps dataset keys to class attributes, avoiding method overwrites.

        :param dataset:
        :return:
        """

        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def title(self) -> str:
        return self.doc_type.replace('_', ' ').title()

    # -- Wrapper Functions -- #
    def get_semantic_count(self) -> int:
        return self.chroma_db.get_semantic_count()

    def get_document_count(self) -> int:
        return self.chroma_db.get_document_count()

    def add_semantic_documents(self, documents: list) -> None:
        self.chroma_db.add_semantic_documents(documents)
