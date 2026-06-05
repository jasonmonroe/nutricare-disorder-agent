# src/huggingface.py

class Huggingface():
    def __init__(self):
        self._api_key = None
        self.api = ''


    def __files(self) -> list:
        return []

    def deploy(self):
        files = self.__files()

        for file in files:
            self._upload(file)

    def _upload(self, file):
        pass

