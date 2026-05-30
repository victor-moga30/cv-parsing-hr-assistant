from sentence_transformers import SentenceTransformer, util


class SemanticMatcher:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def similarity_score(self, cv_text, job_text):
        cv_emb = self.model.encode(str(cv_text), convert_to_tensor=True, normalize_embeddings=True)
        job_emb = self.model.encode(str(job_text), convert_to_tensor=True, normalize_embeddings=True)

        score = util.cos_sim(cv_emb, job_emb).item()
        return round(score * 100, 2)