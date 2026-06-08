# models/mock_embeddings

import random

class MockEmbeddings:
    """
    A local mock embedding class to bypass proxy servers during offline testing.
    NOTE: 💡 Use this for mocking if your OPENAI_API_BASE has been deactivated.
    """
    
    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Instantly returns dummy vectors for all 252 chunks locally
        return [[random.uniform(-1, 1) for _ in range(self.dimensions)] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        # Instantly returns a dummy vector for single search queries
        return [random.uniform(-1, 1) for _ in range(self.dimensions)]
