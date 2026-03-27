from parser import load_text_file, load_all_cvs
from vectorizer import tfidf_scores, sbert_scores
from scorer import apply_threshold, normalize_scores

def run_pipeline(job_path: str, cv_dir: str, threshold: float = 0.3):

    # 1. Chargement
    job_desc = load_text_file(job_path)
    cvs      = load_all_cvs(cv_dir)

    print(f"\n{len(cvs)} CV chargés.\n")

    # 2. Scoring TF-IDF
    scores_tfidf = tfidf_scores(job_desc, cvs)
    scores_tfidf_norm = normalize_scores(scores_tfidf)

    # 3. Scoring SBERT
    scores_sbert = sbert_scores(job_desc, cvs)
    # SBERT produit déjà des valeurs entre -1 et 1, on normalise aussi
    scores_sbert_norm = normalize_scores(scores_sbert)

    # 4. Affichage comparatif
    print(f"{'Candidat':<15} {'TF-IDF':>10} {'SBERT':>10} {'Δ':>8}")
    print("-" * 45)
    for nom in cvs:
        tf  = scores_tfidf_norm.get(nom, 0)
        sb  = scores_sbert_norm.get(nom, 0)
        diff = sb - tf
        print(f"{nom:<15} {tf:>10.3f} {sb:>10.3f} {diff:>+8.3f}")

    # 5. Sélection finale (sur les scores SBERT normalisés)
    resultats = apply_threshold(scores_sbert_norm, threshold)
    print(f"\nPrésélectionnés : {list(resultats['selected'].keys())}")
    print(f"Rejetés         : {list(resultats['rejected'].keys())}")

    return resultats

if __name__ == "__main__":
    run_pipeline("data/job_desc.txt", "data/cvs/", threshold=0.4)