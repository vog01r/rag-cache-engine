# Introduction au RAG (Retrieval-Augmented Generation)

Le RAG est une technique qui combine la recherche d'informations avec la génération de texte. 
Cette approche permet aux modèles de langage d'accéder à des bases de connaissances externes 
pour produire des réponses plus précises et à jour.

## Principe de fonctionnement

1. **Ingestion**: Les documents sont traités et convertis en vecteurs
2. **Indexation**: Les vecteurs sont stockés dans une base vectorielle
3. **Recherche**: Pour une question, on trouve les documents les plus pertinents
4. **Génération**: Le LLM utilise ces documents comme contexte pour répondre

## Avantages du RAG

- Accès à des informations spécifiques et récentes
- Réduction des hallucinations du modèle
- Possibilité de citer les sources
- Mise à jour facile de la base de connaissances
