# Méthodologie RAG (Retrieval-Augmented Generation)

## 📖 Introduction

Le **RAG (Retrieval-Augmented Generation)** est une technique qui combine la recherche d'informations (retrieval) avec la génération de texte par des modèles de langage. Cette approche permet de créer des systèmes de questions-réponses plus précis, à jour et vérifiables.

## 🎯 Problèmes résolus par le RAG

### Limitations des LLM traditionnels
- **Connaissances figées** : Entraînement sur des données avec date de coupure
- **Hallucinations** : Génération d'informations incorrectes ou inventées
- **Manque de spécificité** : Difficultés avec des domaines très spécialisés
- **Pas de sources** : Impossible de vérifier l'origine des informations

### Avantages du RAG
- ✅ **Informations à jour** : Accès à des documents récents
- ✅ **Réduction des hallucinations** : Ancrage sur des sources réelles
- ✅ **Traçabilité** : Citation des sources utilisées
- ✅ **Spécialisation** : Adaptation à des domaines spécifiques
- ✅ **Mise à jour facile** : Ajout de nouveaux documents sans réentraînement

## 🔄 Flux RAG classique

```
┌─────────────────┐
│ 1. INGESTION    │
│                 │
│ Documents  ────▶│ Chunking ────▶ Embeddings ────▶ Vector Store
│ (.pdf, .md)     │ (500 tokens)   (sentence-bert)   (FAISS)
└─────────────────┘

┌─────────────────┐
│ 2. RECHERCHE    │
│                 │
│ Query ────▶ Embedding ────▶ Similarity Search ────▶ Top-K Documents
│ (utilisateur)   (même modèle)    (cosine/L2)        (k=3-5)
└─────────────────┘

┌─────────────────┐
│ 3. GÉNÉRATION   │
│                 │
│ Prompt = Context + Query ────▶ LLM ────▶ Response
│ (documents + question)         (GPT/Claude)  (avec sources)
└─────────────────┘
```

## 🏗️ Composants techniques

### 1. Embeddings (Vectorisation)

**Objectif** : Convertir le texte en représentations numériques capturant le sens sémantique.

**Modèles populaires** :
- `all-MiniLM-L6-v2` : Léger, rapide, 384 dimensions
- `all-mpnet-base-v2` : Plus précis, 768 dimensions
- `text-embedding-ada-002` (OpenAI) : Propriétaire, très performant

**Métriques de similarité** :
- **Cosinus** : Mesure l'angle entre vecteurs (ignore la magnitude)
- **Distance euclidienne (L2)** : Distance géométrique dans l'espace
- **Produit scalaire** : Sensible à la magnitude

### 2. Vector Store (Base vectorielle)

**Objectif** : Stockage efficace et recherche rapide de vecteurs haute dimension.

**Solutions** :
- **FAISS** : Bibliothèque Facebook, très rapide, en mémoire ou sur disque
- **Chroma** : Base vectorielle open-source avec métadonnées
- **Pinecone** : Service cloud spécialisé
- **Weaviate** : Base vectorielle avec GraphQL

**Algorithmes d'indexation** :
- **Flat (brute force)** : Précis mais lent pour grandes collections
- **IVF** : Index par quantification vectorielle
- **HNSW** : Graphes de mondes petits hiérarchiques

### 3. Chunking (Segmentation)

**Objectif** : Diviser les documents longs en segments cohérents pour l'embedding.

**Stratégies** :
- **Par taille fixe** : 500-1000 tokens avec overlap de 10-20%
- **Par structure** : Paragraphes, sections, phrases
- **Sémantique** : Regroupement par similarité thématique
- **Hybride** : Combinaison de plusieurs approches

**Paramètres clés** :
- `chunk_size` : Taille des segments (balance précision/contexte)
- `overlap` : Chevauchement pour préserver la continuité
- `separators` : Délimiteurs prioritaires (\n\n, \n, ., etc.)

### 4. Retrieval (Récupération)

**Objectif** : Trouver les documents les plus pertinents pour une requête.

**Approches** :
- **Dense retrieval** : Similarité vectorielle (embeddings)
- **Sparse retrieval** : Mots-clés (BM25, TF-IDF)
- **Hybride** : Combinaison dense + sparse avec pondération

**Filtrage** :
- **Métadonnées** : Date, auteur, type de document
- **Score de confiance** : Seuil de similarité minimum
- **Diversité** : Maximum Marginal Relevance (MMR)

### 5. Génération

**Objectif** : Produire une réponse naturelle basée sur le contexte récupéré.

**Techniques de prompting** :
```
Système: Tu es un assistant utile qui répond uniquement basé sur le contexte fourni.

Contexte:
[Document 1] Le RAG combine recherche et génération...
[Document 2] FAISS est une bibliothèque vectorielle...

Question: Qu'est-ce que le RAG ?

Instructions:
- Réponds uniquement avec les informations du contexte
- Cite les documents utilisés
- Si l'info n'est pas disponible, dis-le clairement

Réponse:
```

## ⚡ Optimisations et Cache

### Stratégies de cache

#### 1. Cache de réponses
- **Clé** : Hash de la question + paramètres (k, filtres)
- **Valeur** : Réponse complète générée
- **TTL** : 1-24h selon la fraîcheur requise
- **Éviction** : LRU, LFU, ou TTL strict

#### 2. Cache d'embeddings
- **Clé** : Hash du texte
- **Valeur** : Vecteur d'embedding
- **Avantage** : Évite le recalcul pour documents statiques
- **Stockage** : Redis, fichier local, base SQL

#### 3. Cache de recherche
- **Clé** : Hash de la requête + paramètres de recherche
- **Valeur** : Liste des documents récupérés avec scores
- **Utilité** : Pour requêtes similaires ou variantes

### Optimisations de performance

#### Index vectoriel
```python
# FAISS - configuration pour la performance
import faiss

# Index plat (précis, lent pour >10K vecteurs)
index = faiss.IndexFlatL2(dimension)

# Index IVF (rapide, bon compromis)
nlist = 100  # nombre de clusters
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Index HNSW (très rapide, plus de mémoire)
index = faiss.IndexHNSWFlat(dimension, 32)
```

#### Chunking intelligent
```python
# Chunking avec overlap sémantique
def semantic_chunking(text, model, max_size=500, similarity_threshold=0.8):
    sentences = split_sentences(text)
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        if len(current_chunk) >= max_size:
            # Vérifier la cohérence sémantique
            if semantic_similarity(current_chunk, [sentence]) < similarity_threshold:
                chunks.append(current_chunk)
                current_chunk = [sentence]
            else:
                current_chunk.append(sentence)
        else:
            current_chunk.append(sentence)
    
    return chunks
```

## 📊 Métriques et évaluation

### Métriques de récupération
- **Recall@k** : Proportion de documents pertinents dans les k premiers
- **Precision@k** : Proportion de documents pertinents parmi les k récupérés
- **MRR (Mean Reciprocal Rank)** : Position moyenne du premier document pertinent
- **NDCG (Normalized DCG)** : Qualité du classement avec pertinence graduée

### Métriques de génération
- **BLEU** : Comparaison n-grammes avec réponses de référence
- **ROUGE** : Overlap de séquences (résumés)
- **BERTScore** : Similarité sémantique avec BERT
- **Faithfulness** : Fidélité au contexte fourni

### Métriques système
- **Latence** : Temps de réponse (ingestion, recherche, génération)
- **Throughput** : Requêtes par seconde
- **Cache hit rate** : Efficacité du système de cache
- **Coût** : Appels API, compute, stockage

## 🔮 Améliorations avancées

### 1. Recherche hybride
Combinaison de méthodes denses et sparse :
```python
def hybrid_search(query, alpha=0.7):
    # Recherche dense (embeddings)
    dense_results = vector_search(query, k=20)
    
    # Recherche sparse (BM25)
    sparse_results = bm25_search(query, k=20)
    
    # Fusion des scores
    final_results = []
    for doc in all_documents:
        dense_score = get_dense_score(doc, dense_results)
        sparse_score = get_sparse_score(doc, sparse_results)
        
        # Score hybride pondéré
        hybrid_score = alpha * dense_score + (1-alpha) * sparse_score
        final_results.append((doc, hybrid_score))
    
    return sorted(final_results, key=lambda x: x[1], reverse=True)[:k]
```

### 2. Re-ranking
Amélioration du classement avec modèles spécialisés :
```python
from sentence_transformers import CrossEncoder

# Modèle de re-ranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def rerank_documents(query, candidates):
    # Calcul des scores de pertinence
    pairs = [(query, doc['text']) for doc in candidates]
    scores = reranker.predict(pairs)
    
    # Nouveau classement
    for i, doc in enumerate(candidates):
        doc['rerank_score'] = scores[i]
    
    return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
```

### 3. Query expansion
Enrichissement des requêtes pour améliorer la récupération :
```python
def expand_query(query, model):
    # Génération de variantes
    expanded = model.generate(
        f"Générez 3 reformulations de cette question: {query}",
        max_tokens=100
    )
    
    # Recherche avec toutes les variantes
    all_results = []
    for variant in [query] + expanded:
        results = vector_search(variant, k=5)
        all_results.extend(results)
    
    # Déduplication et fusion
    return deduplicate_and_merge(all_results)
```

## 🛠️ Implémentation en production

### Architecture scalable
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Auth Service  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAG Service   │◀──▶│  Vector Store   │◀──▶│     Cache       │
│   (FastAPI)     │    │   (Pinecone)    │    │    (Redis)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM Service   │    │   Monitoring    │    │    Logging      │
│  (OpenAI API)   │    │ (Prometheus)    │    │ (Elasticsearch) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Considérations de sécurité
- **Sanitisation** : Validation et nettoyage des inputs
- **Rate limiting** : Protection contre les abus
- **Authentification** : Contrôle d'accès aux API
- **Chiffrement** : Données sensibles en transit et au repos
- **Audit** : Traçabilité des requêtes et réponses

### Monitoring et observabilité
```python
import time
from prometheus_client import Counter, Histogram, Gauge

# Métriques personnalisées
query_counter = Counter('rag_queries_total', 'Total queries processed')
query_duration = Histogram('rag_query_duration_seconds', 'Query processing time')
cache_hit_rate = Gauge('rag_cache_hit_rate', 'Cache hit rate percentage')

def monitored_rag_query(query):
    start_time = time.time()
    query_counter.inc()
    
    try:
        result = rag_engine.answer(query)
        
        # Mise à jour des métriques
        duration = time.time() - start_time
        query_duration.observe(duration)
        
        if result['cached']:
            cache_hit_rate.inc()
            
        return result
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise
```

Ce guide fournit une base solide pour comprendre et implémenter des systèmes RAG robustes et performants. La démonstration de ce projet illustre ces concepts dans un contexte pratique et extensible.
