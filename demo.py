#!/usr/bin/env python3
"""
Script de démonstration pour le système RAG avec cache.
Montre un exemple complet: ingestion -> vectorisation -> requête -> réponse.
"""

import os
import sys
from pathlib import Path

# Ajout du répertoire src au path Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.embedder import SentenceTransformersEmbedder
from src.vector_store import VectorStore
from src.llm_adapter import MockLLM
from src.rag_engine import RAGEngine
from src.utils import (
    setup_logging, load_config, load_documents_from_directory, 
    chunk_documents, ensure_directory
)


def print_separator(title: str) -> None:
    """Affiche un séparateur avec titre."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title: str) -> None:
    """Affiche un sous-titre."""
    print(f"\n--- {title} ---")


def main():
    """Fonction principale de démonstration."""
    print_separator("DÉMONSTRATION RAG AVEC CACHE")
    print("Ce script démontre un système RAG complet avec:")
    print("- Ingestion de documents")
    print("- Vectorisation avec sentence-transformers")
    print("- Recherche vectorielle avec FAISS")
    print("- Cache intelligent des réponses")
    print("- Génération mockée de réponses LLM")
    
    # 1. Configuration et logging
    print_subsection("1. Configuration")
    setup_logging("INFO")
    config = load_config()
    print(f"✓ Configuration chargée")
    
    # 2. Initialisation des composants
    print_subsection("2. Initialisation des composants")
    
    # Embedder
    embedder = SentenceTransformersEmbedder(
        model_name=config['embedding']['model_name']
    )
    print(f"✓ Embedder initialisé: {config['embedding']['model_name']}")
    
    # Vector store
    vector_store = VectorStore(dimension=config['vector_store']['dimension'])
    print(f"✓ Vector store initialisé (dimension: {config['vector_store']['dimension']})")
    
    # LLM adapter (mock)
    llm_adapter = MockLLM()
    print(f"✓ LLM adapter mocké initialisé")
    
    # RAG Engine
    rag_engine = RAGEngine(
        embedder=embedder,
        vector_store=vector_store,
        llm_adapter=llm_adapter,
        k=config['retrieval']['k'],
        enable_cache=True,
        cache_ttl=config['cache']['ttl_seconds']
    )
    print(f"✓ Moteur RAG initialisé (k={config['retrieval']['k']}, cache activé)")
    
    # 3. Chargement et ingestion des documents
    print_subsection("3. Chargement des documents")
    
    docs_dir = "samples/docs_to_ingest"
    if not os.path.exists(docs_dir) or len(os.listdir(docs_dir)) == 0:
        print(f"⚠️  Répertoire {docs_dir} vide ou non trouvé. Création d'exemples...")
        create_sample_documents()
    
    # Chargement des documents
    documents = load_documents_from_directory(docs_dir)
    print(f"✓ Chargé {len(documents)} documents depuis {docs_dir}")
    
    for doc in documents:
        filename = doc.get('filename', 'inconnu')
        text_length = len(doc.get('text', ''))
        print(f"  - {filename}: {text_length} caractères")
    
    # Chunking des documents
    print_subsection("4. Chunking et ingestion")
    chunked_docs = chunk_documents(
        documents, 
        max_tokens=config['chunking']['max_tokens'],
        overlap=config['chunking']['overlap']
    )
    print(f"✓ Documents divisés en {len(chunked_docs)} chunks")
    
    # Ingestion dans le RAG
    rag_engine.ingest_documents(chunked_docs)
    print(f"✓ {len(chunked_docs)} chunks ingérés dans le système RAG")
    
    # 5. Démonstration des requêtes
    print_separator("DÉMONSTRATION DES REQUÊTES")
    
    # Questions d'exemple
    sample_queries = [
        "Qu'est-ce que le RAG ?",
        "Comment fonctionne la vectorisation ?",
        "Quels sont les avantages du cache ?",
        "Qu'est-ce que FAISS ?",
        "Comment améliorer les performances ?"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print_subsection(f"Requête {i}: {query}")
        
        # Première exécution (sans cache)
        result = rag_engine.answer(query)
        
        print(f"Temps de traitement: {result['processing_time']:.3f}s")
        print(f"Documents trouvés: {len(result['documents'])}")
        print(f"Mise en cache: {'Oui' if not result['cached'] else 'Était déjà en cache'}")
        print(f"Longueur du contexte: {result.get('context_length', 0)} caractères")
        
        # Affichage des sources
        if result['documents']:
            print("\nSources utilisées:")
            for j, doc in enumerate(result['documents'][:3], 1):  # Top 3
                source = doc.get('source', 'Source inconnue')
                distance = doc.get('distance', 0)
                print(f"  {j}. {os.path.basename(source)} (distance: {distance:.3f})")
        
        # Affichage de la réponse (tronquée)
        answer = result['answer']
        if len(answer) > 300:
            answer = answer[:300] + "..."
        print(f"\nRéponse:\n{answer}")
        
        # Test du cache (deuxième exécution de la même requête)
        if i == 1:  # Test cache seulement pour la première requête
            print("\n🔄 Test du cache (même requête)...")
            cached_result = rag_engine.answer(query)
            print(f"Temps avec cache: {cached_result['processing_time']:.3f}s")
            print(f"Récupéré du cache: {'Oui' if cached_result['cached'] else 'Non'}")
    
    # 6. Statistiques finales
    print_separator("STATISTIQUES DU SYSTÈME")
    stats = rag_engine.get_statistics()
    
    print(f"Documents dans le vector store: {stats['vector_store_size']}")
    print(f"Modèle d'embedding: {stats['embedder_model']}")
    print(f"Nombre de documents récupérés (k): {stats['retrieval_k']}")
    print(f"Cache activé: {'Oui' if stats['cache_enabled'] else 'Non'}")
    
    if 'cache_stats' in stats:
        cache_stats = stats['cache_stats']
        print(f"Entrées en cache: {cache_stats['size']}/{cache_stats['max_size']}")
        print(f"TTL du cache: {cache_stats['ttl_seconds']}s")
    
    # 7. Test de sauvegarde (optionnel)
    print_subsection("7. Test de sauvegarde")
    save_dir = "demo_vector_store"
    ensure_directory(save_dir)
    vector_store.save(save_dir)
    print(f"✓ Vector store sauvegardé dans {save_dir}")
    
    print_separator("DÉMONSTRATION TERMINÉE")
    print("🎉 La démonstration s'est terminée avec succès !")
    print("\nProchaines étapes possibles:")
    print("- Examiner le code dans src/")
    print("- Lancer les tests: pytest tests/")
    print("- Lire la documentation: docs/RAG.md")
    print("- Modifier config.yaml pour ajuster les paramètres")
    print("- Ajouter vos propres documents dans samples/docs_to_ingest/")


def create_sample_documents():
    """Crée des documents d'exemple pour la démonstration."""
    ensure_directory("samples/docs_to_ingest")
    
    # Document 1: Introduction au RAG
    doc1_content = """# Introduction au RAG (Retrieval-Augmented Generation)

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
"""

    # Document 2: Vectorisation et embeddings
    doc2_content = """# Vectorisation et Embeddings

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
"""

    # Document 3: Cache et optimisation
    doc3_content = """# Stratégies de Cache et Optimisation

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
"""

    # Sauvegarde des documents
    documents = [
        ("introduction_rag.md", doc1_content),
        ("vectorisation.md", doc2_content),
        ("cache_optimisation.md", doc3_content)
    ]
    
    for filename, content in documents:
        filepath = os.path.join("samples/docs_to_ingest", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Document d'exemple créé: {filename}")


if __name__ == "__main__":
    main()
