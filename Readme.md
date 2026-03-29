# RecrutAI — Système de filtrage automatique de candidatures

> Projet NLP — Filtrage de CV par TF-IDF & Sentence-BERT + Chatbot présélectif via LLM

---

## Aperçu

RecrutAI est un système automatisé de présélection de candidatures en deux étapes :

1. **Filtrage des CV** par similarité sémantique (TF-IDF et Sentence-BERT)
2. **Entretien chatbot** adaptatif via un LLM (LLaMA 3.3 70B / Groq)

Le recruteur humain ne reçoit que les candidats les mieux classés après les deux filtres.

---

## Stack technique

| Composant | Technologie |
|---|---|
| Vectorisation classique | TF-IDF — `scikit-learn` |
| Vectorisation sémantique | SBERT — `paraphrase-multilingual-MiniLM-L12-v2` |
| Chatbot LLM | LLaMA 3.3 70B via API Groq (gratuit) |
| Interface | Streamlit |
| Langage | Python 3.10+ |

---

## Structure du projet

```
projetNLP/
├── data/
│   ├── job_desc.txt        # Fiche de poste
│   └── cvs/
│       ├── cv1.txt         # Profil fort
│       ├── cv2.txt         # Profil moyen
│       ├── cv3.txt         # Profil fort
│       ├── cv4.txt         # Profil faible
│       └── cv5.txt         # Profil moyen
├── src/
│   └── chatbot.py          # Logique du chatbot LLM
├── app.py                  # Application Streamlit
├── test.py                 # Test du pipeline NLP en terminal
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/projetNLP.git
cd projetNLP

# 2. Installer les dépendances
py -m pip install -r requirements.txt
```

`requirements.txt` :
```
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
streamlit>=1.28.0
groq>=0.4.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## Configuration

1. Créez un compte gratuit sur [console.groq.com](https://console.groq.com)
2. Générez une clé API
3. Dans `src/chatbot.py`, remplacez :

```python
client = Groq(api_key="gsk_xxxxxxxxxxxxxxxxxxxxxxxx")
```

> **Ne committez jamais votre clé API.** Ajoutez un fichier `.env` et utilisez `python-dotenv`, ou configurez la variable d'environnement `GROQ_API_KEY`.

---

## Utilisation

### Test rapide en terminal

```bash
py test.py
```

Affiche le tableau comparatif TF-IDF vs SBERT pour tous les CV.

### Lancer l'interface web

```bash
py -m streamlit run app.py
```

Ouvre l'application sur `http://localhost:8501`.

### Workflow

1. Cliquer **"Analyser les CV"** dans la barre latérale
2. Onglet **Filtrage CV** → scores TF-IDF vs SBERT + graphique
3. Ajuster le seuil de présélection avec le slider
4. Onglet **Entretien Chatbot** → sélectionner un candidat → démarrer
5. Répondre aux questions → le chatbot conclut avec un score /10
6. Score final combiné affiché avec recommandation

---

## Score final

```
score_final = 0.4 × score_cv (SBERT) + 0.6 × score_chatbot
```

| Score final | Recommandation |
|---|---|
| ≥ 0.75 | Entretien humain prioritaire |
| 0.60 — 0.74 | Entretien humain recommandé |
| 0.40 — 0.59 | À examiner |
| < 0.40 | Non retenu |

---

## Pourquoi ces choix ?

**TF-IDF** — baseline rapide et interprétable, efficace pour les correspondances exactes de mots-clés.

**SBERT** — capture la sémantique : *"développeur"* et *"ingénieur logiciel"* sont reconnus comme proches. Le modèle multilingue MiniLM supporte le français.

**Groq / LLaMA 3.3** — API gratuite, très rapide, compatible syntaxe OpenAI. Permet un entretien adaptatif au profil de chaque candidat sans frais.

---

## Référence

- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
- [Documentation sentence-transformers](https://www.sbert.net)
- [Documentation Groq API](https://console.groq.com/docs)