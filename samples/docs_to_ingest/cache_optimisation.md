# Stratégies de Cache et Optimisation

Le cache est crucial pour optimiser les performances d'un système RAG en production.

## Types de cache

### Cache de réponses
- Stocke les réponses complètes pour des questions identiques
- TTL (Time To Live) configurable
- Évite les appels répétés au LLM

### Cache de vecteurs
- Stocke les embeddings calculés
- Évite le recalcul pour des textes identiques
- Particulièrement utile pour les documents statiques

## Algorithmes d'éviction

- **LRU (Least Recently Used)**: Supprime les entrées les moins récemment utilisées
- **LFU (Least Frequently Used)**: Supprime les entrées les moins fréquemment utilisées
- **TTL**: Supprime les entrées expirées

## Optimisations supplémentaires

- Indexation hiérarchique
- Filtrage par métadonnées
- Recherche hybride (dense + sparse)
- Quantification des vecteurs
