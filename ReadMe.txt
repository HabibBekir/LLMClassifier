##Test technique Santévet - Classification de messages clients

Prototype de classification avec extraction d'entités (nom_animal, espece, numero_contrat) via LLM local (Ollama).

#Lancer le projet

1. Installer Ollama 
2. ollama pull llama3.1:8b
3. pip install -r requirements.txt
4. python LLMClassifier.py

Le script lit messages.json et écrit resultats.json.

#Fichiers

- LLMClassifier.py : code source
- messages.json : données d'entrée
- resultats.json : sortie produite par le script
- requirements.txt

#Choix techniques

- Ollama + llama3.1:8b : pas de clé API, données qui restent en local, 100% gratuit.
- format="json" + temperature=0 : sortie JSON déterministe et fiable.
- Validation Pydantic : la sortie du LLM est validée contre un schéma strict (catégorie dans une liste fermée, espece typée, etc.). Si le LLM hallucine une catégorie inexistante, on attrape l'erreur proprement et on bascule en "autre" pour ne pas casser le pipeline.
- Few-shot dans le prompt : règles explicites pour les cas frontaliers (décès + résiliation = resiliation, relance facture = remboursement) + listes de mots interdits comme nom_animal pour éviter que le LLM prenne "chiot" ou "berger australien" pour un nom propre.
