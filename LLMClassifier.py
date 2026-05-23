import json
import logging
import ollama
from pydantic import BaseModel, ValidationError
from typing import Optional, Literal


CATEGORIES = ["sinistre", "resiliation", "question_contrat", "demande_remboursement", "autre"]
ESPECES = ["chien", "chat", "autre"]

SYSTEM_PROMPT = """Tu es un assistant qui classifie des messages clients pour une assurance santé animale.
Tu réponds UNIQUEMENT avec un objet JSON, sans texte avant ni après.

Format attendu :
{
  "categorie": "<sinistre | resiliation | question_contrat | demande_remboursement | autre>",
  "entites": {
    "nom_animal": "<nom propre OU null>",
    "espece": "<chien | chat | autre | null>",
    "numero_contrat": "<SAV-AAAA-XXXXX OU null>"
  }
}

Définition des catégories
- sinistre : accident, maladie, hospitalisation, urgence vétérinaire en cours
- resiliation : demande d'arrêt du contrat (décès, don de l'animal, autre raison)
- question_contrat : question sur les garanties, formules, délais, couvertures
- demande_remboursement : envoi de facture ou relance d'un remboursement
- autre : changement d'adresse, ajout d'animal, réclamation technique, remerciement

Règles de classification
- Décès + résiliation = resiliation (pas sinistre)
- Réclamation + résiliation = resiliation
- Relance de facture = demande_remboursement

Règles d'extraction d'entités

nom_animal: SEUL un prénom/nom propre est valide. En cas de doute, mettre null.

INTERDIT comme nom_animal (mettre null à la place) :
- Mots génériques : "chien", "chienne", "chiot", "chat", "chatte", "chaton", "matou", "compagnon", "animal", "bête"
- Races : "berger australien", "bouledogue français", "labrador", "siamois", "british shorthair", "golden retriever"
- Pronoms : "il", "elle", "le mien"

espece: Tu extrais UNIQUEMENT si un mot du message indique clairement l'espèce.
Mots-clés autorisés :
- chat: "chat", "chatte", "chaton", "matou", "félin"
- chien: "chien", "chienne", "chiot", "canin"
- autre: "furet", "lapin", "perroquet", "chinchilla", "cobaye", "rongeur", "NAC"
Si AUCUN de ces mots n'apparaît dans le message, mettre null.

numero_contrat: Format strict : SAV-AAAA-XXXXX (exemple : SAV-2023-78542). Si aucun numéro de ce format, mettre null.
"""


class Entites(BaseModel):
    nom_animal: Optional[str]
    espece: Optional[Literal["chien", "chat", "autre"]]
    numero_contrat: Optional[str]


class ClassificationResponse(BaseModel):
    categorie: Literal["sinistre", "resiliation", "question_contrat", "demande_remboursement", "autre"]
    entites: Entites


def classify_message(message, model):
    """Classifie un message et retourne un dict avec id, categorie, entites."""
    if message.get("sujet") is not None:
        sujet = message.get("sujet")
    else:
        sujet = "pas de sujet"
    user_prompt = f"Sujet : {sujet}\nMessage : {message['contenu']}"

    response = ollama.chat(model=model,messages=[{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": user_prompt},],format="json",options={"temperature": 0},)

    message_content= response["message"]["content"]
    parsed = ClassificationResponse.model_validate_json(message_content)
    data = parsed.model_dump()

    categorie = data["categorie"] 
    entites = data["entites"]

    return {
        "id": message["id"],
        "categorie": categorie,
        "entites": {
            "nom_animal": entites["nom_animal"],
            "espece": entites["espece"],
            "numero_contrat": entites["numero_contrat"],
        },
    }


 # Main script 
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

with open("messages.json", encoding="utf-8") as f:
    messages = json.load(f)["messages"]

logging.info("Classification de %d messages", len(messages))

results = []
for message in messages:
    try:
        res = classify_message(message, model="llama3.1:8b")
        results.append(res)
        logging.info("[%d] %s", res["id"], res["categorie"])
    except (json.JSONDecodeError, KeyError, ValidationError) as e:
        logging.error("Erreur sur le message %d : %s", message["id"], e)
        results.append({
            "id": message["id"],
            "categorie": "autre",
            "entites": {"nom_animal": None, "espece": None, "numero_contrat": None},
        })

with open("resultats.json", "w", encoding="utf-8") as f:
    json.dump({"resultats": results}, f, ensure_ascii=False, indent=2)

logging.info("Terminé. Résultats dans resultats.json")


