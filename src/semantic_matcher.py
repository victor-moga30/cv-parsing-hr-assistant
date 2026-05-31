from sentence_transformers import SentenceTransformer, util
import re


class SemanticMatcher:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def _clean_text(self, text):
        text = str(text)
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_into_chunks(self, text, max_words=80):
        text = self._clean_text(text)

        parts = re.split(r"(?<=[.!?])\s+| - |\u2022", text)

        chunks = []
        current = []

        for part in parts:
            words = part.strip().split()
            if not words:
                continue

            if len(current) + len(words) <= max_words:
                current.extend(words)
            else:
                if current:
                    chunks.append(" ".join(current))
                current = words

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _cosine_to_percent(self, cosine):
        """
        Pentru sentence embeddings, cosine-ul brut nu trebuie citit direct ca procent.
        Transformam scorul intr-un procent mai realist pentru dashboard.
        """
        if cosine <= 0.20:
            return cosine * 100

        if cosine >= 0.65:
            return 100.0

        return 40.0 + ((cosine - 0.20) / (0.65 - 0.20)) * 60.0

    def similarity_score(self, cv_text, job_text):
        cv_chunks = self._split_into_chunks(cv_text)
        job_chunks = self._split_into_chunks(job_text)

        if not cv_chunks or not job_chunks:
            return 0.0

        cv_embeddings = self.model.encode(
            cv_chunks,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        job_embeddings = self.model.encode(
            job_chunks,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        similarity_matrix = util.cos_sim(job_embeddings, cv_embeddings)

        best_scores = []

        for i in range(len(job_chunks)):
            best_match_for_requirement = similarity_matrix[i].max().item()
            best_scores.append(best_match_for_requirement)

        average_best_score = sum(best_scores) / len(best_scores)

        semantic_percent = self._cosine_to_percent(average_best_score)

        return round(max(0.0, min(100.0, semantic_percent)), 2)