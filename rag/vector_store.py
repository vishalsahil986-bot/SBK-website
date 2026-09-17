from typing import List, Optional

from pinecone import Pinecone

from config.settings import settings
from utils.logger import logger


class PineconeVectorStore:
    def __init__(self):
        self.api_key = settings.pinecone_api_key
        self.index_name = settings.pinecone_index_name

        if not self.api_key:
            raise RuntimeError("PINECONE_API_KEY is not configured")

        self.client = Pinecone(
            api_key=self.api_key
        )

        self.index = self.client.Index(
            self.index_name
        )

        logger.info(
            f"Pinecone vector store connected "
            f"(index={self.index_name})"
        )

    def query(
        self,
        embedding: List[float],
        top_k: Optional[int] = None,
    ) -> List[dict]:

        if not embedding:
            return []

        top_k = top_k or settings.rag_top_k

        try:
            result = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
            )

            matches = []

            for match in result.matches:
                metadata = match.metadata or {}

                text = metadata.get("text", "")
                source = metadata.get("source", "")

                if not text:
                    continue

                matches.append(
                    {
                        "text": text,
                        "source": source,
                        "score": float(match.score),
                    }
                )

            logger.info(
                f"Retrieved {len(matches)} chunks "
                f"from Pinecone index '{self.index_name}'"
            )

            return matches

        except Exception as exc:
            logger.exception(
                f"Pinecone query failed: {exc}"
            )
            return []


vector_store = PineconeVectorStore()