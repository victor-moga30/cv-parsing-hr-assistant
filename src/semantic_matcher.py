import re
from collections import OrderedDict
from typing import List, Tuple

from sentence_transformers import (
    SentenceTransformer,
    util,
)


class SemanticMatcher:
    """
    Sentence-BERT matcher shared by the runtime,
    optimiser and evaluator.

    Each job chunk is compared with all CV chunks.
    The best CV match for each job chunk is retained,
    and those best similarities are averaged.
    """

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        max_words_per_chunk: int = 80,
        cache_size: int = 128,
    ):
        if max_words_per_chunk <= 0:
            raise ValueError(
                "max_words_per_chunk must be "
                "greater than zero."
            )

        if cache_size < 0:
            raise ValueError(
                "cache_size cannot be negative."
            )

        self.model = SentenceTransformer(
            model_name
        )

        self.max_words_per_chunk = (
            max_words_per_chunk
        )

        self.cache_size = cache_size

        self._embedding_cache = (
            OrderedDict()
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        text = str(text).replace(
            "\x00",
            " ",
        )

        text = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"[ \t\f\v]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n+",
            "\n",
            text,
        )

        return text.strip()

    def _split_into_chunks(
        self,
        text: str,
    ) -> List[str]:
        """
        Split text into chunks that never exceed
        max_words_per_chunk.
        """
        cleaned_text = self._clean_text(
            text
        )

        if not cleaned_text:
            return []

        parts = re.split(
            r"(?<=[.!?])\s+"
            r"|\n+"
            r"|\s[-–—]\s"
            r"|\u2022"
            r"|●",
            cleaned_text,
        )

        chunks = []
        current_words = []

        for part in parts:
            remaining_words = (
                part.strip().split()
            )

            while remaining_words:
                available_space = (
                    self.max_words_per_chunk
                    - len(current_words)
                )

                current_words.extend(
                    remaining_words[
                        :available_space
                    ]
                )

                remaining_words = (
                    remaining_words[
                        available_space:
                    ]
                )

                if (
                    len(current_words)
                    == self.max_words_per_chunk
                ):
                    chunks.append(
                        " ".join(
                            current_words
                        )
                    )

                    current_words = []

        if current_words:
            chunks.append(
                " ".join(
                    current_words
                )
            )

        return chunks

    def _encode_text(
        self,
        text: str,
    ) -> Tuple[List[str], object]:
        """
        Encode one document and keep a small LRU
        cache of recent embeddings.
        """
        cleaned_text = self._clean_text(
            text
        )

        if not cleaned_text:
            return [], None

        if (
            cleaned_text
            in self._embedding_cache
        ):
            (
                chunks,
                embeddings,
            ) = self._embedding_cache.pop(
                cleaned_text
            )

            self._embedding_cache[
                cleaned_text
            ] = (
                chunks,
                embeddings,
            )

            return (
                chunks,
                embeddings,
            )

        chunks = self._split_into_chunks(
            cleaned_text
        )

        if not chunks:
            return [], None

        embeddings = self.model.encode(
            chunks,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if self.cache_size > 0:
            self._embedding_cache[
                cleaned_text
            ] = (
                chunks,
                embeddings,
            )

            while (
                len(self._embedding_cache)
                > self.cache_size
            ):
                self._embedding_cache.popitem(
                    last=False
                )

        return (
            chunks,
            embeddings,
        )

    def clear_cache(self) -> None:
        self._embedding_cache.clear()

    def similarity_score(
        self,
        cv_text: str,
        job_text: str,
    ) -> float:
        (
            cv_chunks,
            cv_embeddings,
        ) = self._encode_text(
            cv_text
        )

        (
            job_chunks,
            job_embeddings,
        ) = self._encode_text(
            job_text
        )

        if (
            not cv_chunks
            or not job_chunks
            or cv_embeddings is None
            or job_embeddings is None
        ):
            return 0.0

        similarity_matrix = util.cos_sim(
            job_embeddings,
            cv_embeddings,
        )

        best_scores = [
            similarity_matrix[
                index
            ].max().item()
            for index in range(
                len(job_chunks)
            )
        ]

        average_best_score = (
            sum(best_scores)
            / len(best_scores)
        )

        bounded_score = max(
            0.0,
            min(
                1.0,
                average_best_score,
            ),
        )

        return round(
            bounded_score * 100.0,
            2,
        )