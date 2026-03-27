from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ── Chargement ────────────────────────────────────────────────────────────────
def load_txt(path):
    with open(path, encoding='utf-8') as f:
        return f.read().strip()

job_desc = load_txt("data/job_desc.txt")
cvs = {p.stem: load_txt(p) for p in Path("data/cvs").glob("*.txt")}

print(f"Fiche de poste chargée.")
print(f"{len(cvs)} CV chargés : {list(cvs.keys())}\n")

# ── TF-IDF ────────────────────────────────────────────────────────────────────
corpus = [job_desc] + list(cvs.values())
noms   = list(cvs.keys())

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
matrix     = vectorizer.fit_transform(corpus)

job_vec    = matrix[0]
cv_vecs    = matrix[1:]
scores_tfidf = cosine_similarity(job_vec, cv_vecs).flatten()

# ── SBERT ─────────────────────────────────────────────────────────────────────
print("Chargement du modèle SBERT (première fois = téléchargement)...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embeddings   = model.encode([job_desc] + list(cvs.values()))
job_emb      = embeddings[0].reshape(1, -1)
cv_embs      = embeddings[1:]
scores_sbert = cosine_similarity(job_emb, cv_embs).flatten()

# ── Normalisation 0-1 ─────────────────────────────────────────────────────────
def normalize(scores):
    mn, mx = scores.min(), scores.max()
    return (scores - mn) / (mx - mn) if mx != mn else scores

tfidf_norm = normalize(scores_tfidf)
sbert_norm = normalize(scores_sbert)

# ── Affichage comparatif ──────────────────────────────────────────────────────
SEUIL = 0.4

print(f"\n{'Candidat':<12} {'TF-IDF':>10} {'SBERT':>10} {'Δ':>8}  {'Statut'}")
print("─" * 58)
for i, nom in enumerate(noms):
    tf   = tfidf_norm[i]
    sb   = sbert_norm[i]
    diff = sb - tf
    statut = "✓ Présélectionné" if sb >= SEUIL else "✗ Rejeté"
    print(f"{nom:<12} {tf:>10.3f} {sb:>10.3f} {diff:>+8.3f}  {statut}")

selectionnes = [noms[i] for i in range(len(noms)) if sbert_norm[i] >= SEUIL]
print(f"\nRésultat final : {len(selectionnes)}/{len(noms)} candidats présélectionnés")
print(f"→ {selectionnes}")