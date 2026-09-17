from typing import List

from pinecone import Pinecone

from config.settings import settings
from utils.logger import logger


class EmbeddingService:
    def __init__(self):
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not configured")

        self.model = settings.embedding_model_name
        self.dimension = settings.embedding_dimension

        self.client = Pinecone(
            api_key=settings.pinecone_api_key
        )

        logger.info(
            f"Embedding service initialized "
            f"(model={self.model}, dimension={self.dimension})"
        )

    def _embed(
        self,
        texts: List[str],
        input_type: str,
    ) -> List[List[float]]:
        if not texts:
            return []

        response = self.client.inference.embed(
            model=self.model,
            inputs=texts,
            parameters={
                "input_type": input_type,
                "truncate": "END",
                "dimension": self.dimension,
            },
        )

        return [
            list(item.values)
            for item in response.data
        ]

    def embed_text(self, text: str) -> List[float]:
        """
        Create an embedding for a user's search query.
        """
        if not text or not text.strip():
            return []

        return self._embed(
            [text.strip()],
            "query",
        )[0]

    def embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Create embeddings for documents/passages.

        This is included for compatibility, although the chatbot
        currently only needs embed_text() because your portfolio
        documents are already stored in Pinecone.
        """
        clean_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not clean_texts:
            return []

        return self._embed(
            clean_texts,
            "passage",
        )


embedding_service = EmbeddingService()