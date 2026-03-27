# -*- coding: utf-8 -*-
import streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
sys.path.insert(0, "src")
from chatbot import chat, reset_history

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RecrutAI", page_icon="🤖", layout="wide")
st.title("RecrutAI — Système de filtrage automatique")

# ── Fonctions utilitaires ─────────────────────────────────────────────────────
def load_txt(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()

def load_cvs(cv_dir):
    return {p.stem: load_txt(p) for p in Path(cv_dir).glob("*.txt")}

def normalize(scores):
    mn, mx = scores.min(), scores.max()
    return (scores - mn) / (mx - mn) if mx != mn else scores

def score_cvs(job_desc, cvs):
    noms   = list(cvs.keys())
    corpus = [job_desc] + list(cvs.values())

    # TF-IDF
    vec    = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    matrix = vec.fit_transform(corpus)
    tfidf  = normalize(cosine_similarity(matrix[0], matrix[1:]).flatten())

    # SBERT
    model  = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embs   = model.encode(corpus)
    sbert  = normalize(cosine_similarity(embs[0:1], embs[1:]).flatten())

    return {noms[i]: {"tfidf": round(float(tfidf[i]),3),
                      "sbert": round(float(sbert[i]),3)}
            for i in range(len(noms))}

# ── Sidebar — configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    seuil = st.slider("Seuil de présélection", 0.0, 1.0, 0.4, 0.05)
    st.markdown("---")
    if st.button("Analyser les CV"):
        with st.spinner("Analyse en cours..."):
            job_desc = load_txt("data/job_desc.txt")
            cvs      = load_cvs("data/cvs")
            st.session_state["scores"]   = score_cvs(job_desc, cvs)
            st.session_state["cvs"]      = cvs
            st.session_state["job_desc"] = job_desc
        st.success("Analyse terminée !")

# ── Onglets ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Filtrage CV", "Entretien Chatbot"])

# ── Tab 1 : résultats du scoring ──────────────────────────────────────────────
with tab1:
    if "scores" not in st.session_state:
        st.info("Cliquez sur 'Analyser les CV' dans la barre latérale.")
    else:
        scores   = st.session_state["scores"]
        job_desc = st.session_state["job_desc"]

        st.subheader("Fiche de poste")
        st.text(job_desc)
        st.markdown("---")
        st.subheader("Résultats du scoring")

        selectionnes = []
        for nom, s in sorted(scores.items(),
                             key=lambda x: x[1]["sbert"], reverse=True):
            statut = "✅ Présélectionné" if s["sbert"] >= seuil else "❌ Rejeté"
            if s["sbert"] >= seuil:
                selectionnes.append(nom)
            col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
            col1.write(f"**{nom}**")
            col2.metric("TF-IDF", s["tfidf"])
            col3.metric("SBERT",  s["sbert"])
            col4.write(statut)

        st.markdown("---")
        st.success(f"{len(selectionnes)} candidat(s) présélectionné(s) : "
                   f"{', '.join(selectionnes)}")
        # Graphique comparatif TF-IDF vs SBERT
        st.markdown("---")
        st.subheader("Comparaison TF-IDF vs SBERT")

        import matplotlib.pyplot as plt
        import numpy as np

        noms   = list(scores.keys())
        tfidf  = [scores[n]["tfidf"] for n in noms]
        sbert  = [scores[n]["sbert"] for n in noms]

        x      = np.arange(len(noms))
        width  = 0.35

        fig, ax = plt.subplots(figsize=(8, 4))
        bars1 = ax.bar(x - width/2, tfidf, width,
                       label="TF-IDF", color="#4C9BE8")
        bars2 = ax.bar(x + width/2, sbert, width,
                       label="SBERT",  color="#2ECC71")

        # Ligne de seuil
        ax.axhline(y=seuil, color="red", linestyle="--",
                   linewidth=1.2, label=f"Seuil ({seuil})")

        # Valeurs au-dessus des barres
        for bar in bars1 + bars2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(noms, rotation=15)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score normalisé")
        ax.set_title("Score de similarité — TF-IDF vs SBERT")
        ax.legend()
        plt.tight_layout()

        st.pyplot(fig)

# ── Tab 2 : chatbot ───────────────────────────────────────────────────────────
with tab2:
    if "scores" not in st.session_state:
        st.info("Lancez d'abord l'analyse des CV.")
    else:
        scores = st.session_state["scores"]
        cvs    = st.session_state["cvs"]

        selectionnes = [n for n, s in scores.items() if s["sbert"] >= seuil]

        if not selectionnes:
            st.warning("Aucun candidat présélectionné. Baissez le seuil.")
        else:
            candidat = st.selectbox("Choisir un candidat", selectionnes)

            if st.button("Démarrer / Réinitialiser l'entretien"):
                st.session_state["history"] = reset_history()
                st.session_state["candidat"] = candidat

            if "history" in st.session_state:
                # Affichage de l'historique
                for msg in st.session_state["history"]:
                    role = "🤖 Recruteur IA" if msg["role"] == "assistant" \
                           else f"👤 {candidat}"
                    st.markdown(f"**{role}** : {msg['content']}")

                # Premier message automatique du bot
                if len(st.session_state["history"]) == 0:
                    with st.spinner("Le recruteur prépare la première question..."):
                        premiere = chat(
                            [],
                            cvs[candidat],
                            st.session_state["job_desc"]
                        )
                    st.session_state["history"].append(
                        {"role": "assistant", "content": premiere}
                    )
                    st.rerun()

                # Saisie candidat
                reponse = st.chat_input("Votre réponse...")
                if reponse:
                    st.session_state["history"].append(
                        {"role": "user", "content": reponse}
                    )
                    with st.spinner("..."):
                        bot_reply = chat(
                            st.session_state["history"],
                            cvs[candidat],
                            st.session_state["job_desc"]
                        )
                    st.session_state["history"].append(
                        {"role": "assistant", "content": bot_reply}
                    )
                    st.rerun()
                    # Extraction automatique du score /10 depuis la réponse
                    import re
                    match = re.search(r'(\d+)\s*/\s*10', bot_reply)
                    if match:
                        score_chat = int(match.group(1)) / 10
                        score_cv   = scores[candidat]["sbert"]
                        score_final = round(0.4 * score_cv + 0.6 * score_chat, 3)
                        st.session_state["score_final"] = score_final

            # Affichage du score final si disponible
            if "score_final" in st.session_state:
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Score CV (SBERT)",
                            st.session_state["scores"][candidat]["sbert"])
                col2.metric("Score entretien",
                            round(st.session_state.get("score_final",0) * 0.6 /
                                  0.6, 3))
                col3.metric("Score final combiné",
                            st.session_state["score_final"])
                if st.session_state["score_final"] >= 0.6:
                    st.success("Candidat recommandé pour un entretien humain.")
                else:
                    st.warning("Candidat non retenu.")