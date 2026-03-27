# -*- coding: utf-8 -*-
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Load from environment variable

def build_system_prompt(cv_text: str, job_desc: str) -> str:
    return f"""Tu es un recruteur IA professionnel. Tu mènes un entretien
présélectif en français pour le poste suivant :

FICHE DE POSTE :
{job_desc}

CV DU CANDIDAT :
{cv_text}

Instructions :
- Pose UNE seule question à la fois, courte et précise.
- Commence par te présenter et poser une question d'échauffement.
- Concentre-toi sur les points faibles ou manquants du CV par rapport au poste.
- Après 4 à 5 échanges, conclus avec un score /10 et une recommandation.
- Reste professionnel, bienveillant et concis.
- Réponds TOUJOURS en français."""


def chat(history: list, cv_text: str, job_desc: str) -> str:
    system_prompt = build_system_prompt(cv_text, job_desc)
    messages = [{"role": "system", "content": system_prompt}] + history

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # modèle gratuit et très performant
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content


def reset_history() -> list:
    return []