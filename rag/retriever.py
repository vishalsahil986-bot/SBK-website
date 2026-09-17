import re
from typing import List

from rag.embeddings import embedding_service
from rag.vector_store import vector_store
from config.settings import settings
from utils.logger import logger


class RAGRetriever:

    STOP_WORDS = {
        "what", "which", "who", "how", "when", "where",
        "has", "have", "had", "does", "did", "do",
        "is", "are", "was", "were",
        "the", "a", "an", "and", "or", "of", "to",
        "in", "on", "for", "with",
        "vishal", "sahil",
        "tell", "me", "about",
    }

    def _keywords(self, text: str) -> set:
        words = re.findall(
            r"[a-zA-Z0-9+#.]+",
            text.lower(),
        )

        return {
            word
            for word in words
            if len(word) > 2
            and word not in self.STOP_WORDS
        }

    def _rerank(
        self,
        query: str,
        matches: List[dict],
    ) -> List[dict]:

        query_keywords = self._keywords(query)

        for match in matches:
            text = match.get("text", "")
            text_keywords = self._keywords(text)

            overlap = len(
                query_keywords.intersection(text_keywords)
            )

            vector_score = float(
                match.get("score", 0)
            )

            # Semantic similarity remains primary,
            # keyword relevance gives a small boost.
            match["_final_score"] = (
                vector_score
                + (overlap * 0.08)
            )

        return sorted(
            matches,
            key=lambda item: item["_final_score"],
            reverse=True,
        )

    def retrieve(self, query: str) -> str:

        if not query or not query.strip():
            return ""

        try:
            # Convert question to embedding
            query_embedding = embedding_service.embed_text(
                query.strip()
            )

            if not query_embedding:
                logger.warning(
                    "Could not generate query embedding"
                )
                return ""

            # Search more candidates internally.
            # Only the best RAG_TOP_K are sent to Gemini.
            candidate_count = max(
                settings.rag_top_k * 3,
                8,
            )

            matches = vector_store.query(
                embedding=query_embedding,
                top_k=candidate_count,
            )

            if not matches:
                logger.info(
                    f"No RAG results found for: {query}"
                )
                return ""

            # Local reranking - no LLM/token cost
            matches = self._rerank(
                query,
                matches,
            )

            # Send only configured number to Gemini
            matches = matches[
                :settings.rag_top_k
            ]

            context_parts: List[str] = []

            for index, match in enumerate(
                matches,
                start=1,
            ):
                text = match.get(
                    "text",
                    "",
                ).strip()

                source = match.get(
                    "source",
                    "",
                ).strip()

                vector_score = match.get(
                    "score",
                    0,
                )

                final_score = match.get(
                    "_final_score",
                    vector_score,
                )

                if not text:
                    continue

                chunk = (
                    f"[Knowledge {index}]\n"
                    f"{text}"
                )

                if source:
                    chunk += (
                        f"\nSource: {source}"
                    )

                context_parts.append(chunk)

                logger.info(
                    f"RAG result {index}: "
                    f"vector={vector_score:.4f}, "
                    f"reranked={final_score:.4f}, "
                    f"source={source or 'unknown'}"
                )

            return "\n\n".join(
                context_parts
            )

        except Exception as exc:
            logger.exception(
                f"RAG retrieval failed: {exc}"
            )
            return ""


retriever = RAGRetriever()