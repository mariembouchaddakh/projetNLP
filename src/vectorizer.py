from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# ── Approche 1 : TF-IDF ──────────────────────────────────────────────────────

def tfidf_scores(job_desc: str, cvs: dict[str, str]) -> dict[str, float]:
    """
    Vectorise la fiche de poste + tous les CV avec TF-IDF,
    puis calcule la similarité cosinus de chaque CV avec le poste.
    """
    corpus = [job_desc] + list(cvs.values())
    noms   = list(cvs.keys())

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrammes + bigrammes
        stop_words=None,      # gardez les mots français (pas de liste intégrée)
        max_features=5000
    )
    matrix = vectorizer.fit_transform(corpus)

    # Ligne 0 = fiche de poste, lignes 1..N = CV
    job_vec = matrix[0]
    cv_vecs = matrix[1:]

    scores = cosine_similarity(job_vec, cv_vecs).flatten()
    return dict(zip(noms, scores.tolist()))


# ── Approche 2 : Sentence-BERT ───────────────────────────────────────────────

def sbert_scores(job_desc: str, cvs: dict[str, str]) -> dict[str, float]:
    """
    Encode la fiche de poste + CV avec un modèle SBERT,
    puis calcule la similarité cosinus dans l'espace d'embeddings.
    """
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # Ce modèle supporte le français — important à mentionner dans l'exposé

    textes = [job_desc] + list(cvs.values())
    noms   = list(cvs.keys())

    embeddings = model.encode(textes, convert_to_numpy=True)

    job_emb = embeddings[0].reshape(1, -1)
    cv_embs = embeddings[1:]

    scores = cosine_similarity(job_emb, cv_embs).flatten()
    return dict(zip(noms, scores.tolist()))