from sentence_transformers import SentenceTransformer, util


class SemanticMatcher:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def similarity_score(self, cv_text, job_text):
        cv_emb = self.model.encode(
            str(cv_text),
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        job_emb = self.model.encode(
            str(job_text),
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        cosine = util.cos_sim(cv_emb, job_emb).item()

        # Cosine similarity poate fi intre -1 si 1.
        # Pentru dashboard vrem procent valid intre 0 si 100.
        normalized_score = max(0.0, min(1.0, cosine))

        return round(normalized_score * 100, 2)