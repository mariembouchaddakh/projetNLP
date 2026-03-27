def apply_threshold(scores: dict[str, float],
                    threshold: float = 0.3) -> dict:
    """
    Sépare les candidats en deux groupes selon le seuil.
    Retourne un dict avec 'selected' et 'rejected', triés par score.
    """
    selected = {k: v for k, v in scores.items() if v >= threshold}
    rejected = {k: v for k, v in scores.items() if v < threshold}

    return {
        "selected": dict(sorted(selected.items(),
                                key=lambda x: x[1], reverse=True)),
        "rejected": dict(sorted(rejected.items(),
                                key=lambda x: x[1], reverse=True))
    }

def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Ramène les scores entre 0 et 1 (utile pour SBERT dont les valeurs
    sont déjà dans cet intervalle, mais pas toujours pour TF-IDF)."""
    if not scores:
        return {}
    max_s = max(scores.values())
    min_s = min(scores.values())
    if max_s == min_s:
        return {k: 1.0 for k in scores}
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}