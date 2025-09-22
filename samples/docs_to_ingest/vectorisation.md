# Vectorisation et Embeddings

La vectorisation est le processus de conversion de texte en représentations numériques 
que les machines peuvent traiter efficacement.

## Qu'est-ce qu'un embedding ?

Un embedding est un vecteur de nombres réels qui représente le sens sémantique d'un texte.
Les textes similaires ont des embeddings proches dans l'espace vectoriel.

## Modèles d'embedding

- **Sentence-BERT**: Optimisé pour les phrases et paragraphes
- **OpenAI Ada**: Modèle propriétaire performant
- **all-MiniLM-L6-v2**: Modèle compact et efficace (utilisé dans cette démo)

## Métriques de similarité

- Distance euclidienne
- Similarité cosinus
- Distance de Manhattan

FAISS utilise généralement la distance L2 (euclidienne) pour la recherche rapide.
