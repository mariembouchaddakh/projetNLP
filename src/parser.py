import os
from pathlib import Path

def load_text_file(filepath: str) -> str:
    """Charge un fichier .txt et retourne son contenu."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_all_cvs(cv_dir: str) -> dict[str, str]:
    """
    Charge tous les fichiers .txt dans un dossier.
    Retourne un dict {nom_fichier: texte}.
    """
    cvs = {}
    for path in Path(cv_dir).glob("*.txt"):
        cvs[path.stem] = load_text_file(path)
    return cvs