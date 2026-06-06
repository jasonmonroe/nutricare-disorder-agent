# src/huggingface.py

# https://huggingface.co/
# Documents: https://huggingface.co/docs

from huggingface_hub import HfApi
from src.config import HF_TOKEN, DOCUMENT_ZIP, HF_REPO_ID

"""
Section 3: Getting Your App Live on Hugging Face Docker Spaces
"""
class Huggingface:
    def __init__(self) -> None:
        self.api = HfApi(token=HF_TOKEN)
        self._repo_id = HF_REPO_ID
        self._repo_type = 'space'

    def deploy(self) -> None:

        self.api.upload_file(
            path_or_fileobj="requirements.txt",
            path_in_repo="requirements.txt",
            repo_id=self._repo_id,
            repo_type=self._repo_type,
            commit_description="Ran requirements cell to create file and uploading it to HuggingFace.co"
        )

        self.api.upload_file(
            path_or_fileobj="app.py",
            path_in_repo="app.py",
            repo_id=self._repo_id,
            repo_type=self._repo_type,
            commit_description="Ran app.py cell to create file and uploading it to HuggingFace.co"
        )

        self.api.upload_folder(
            folder_path="vectorstore/nutritional_db",
            path_in_repo="nutritional_db",
            repo_id=self._repo_id,
            repo_type=self._repo_type,

            # Optional: Add a short commit message
            commit_description="Pushing nutritional database from vector storage and uploading it to HuggingFace.co"
        )

        # Upload Dockerfile
        self.api.upload_file(
            path_or_fileobj="Dockerfile",
            path_in_repo="Dockerfile",
            repo_id=self._repo_id, # Replace it with your username and space name
            repo_type=self._repo_type,
            commit_description="Ran Dockerfile cell to create file and uploading it to HuggingFace.co"
        )

        # Upload Documents
        self.api.upload_file(
            path_or_fileobj=DOCUMENT_ZIP,
            path_in_repo=DOCUMENT_ZIP,
            repo_id=self._repo_id,
            repo_type=self._repo_type,
            commit_message="Add data zip file for Dockerfile to unzip.",
            commit_description="Uploading the compressed medical reference data to be unzipped during the Docker image build process.",
        )
