# RAG Caching Demo

Une démonstration pédagogique et reproductible de la méthodologie **RAG (Retrieval-Augmented Generation)** avec système de cache intelligent.

## 🎯 Objectif

Ce projet illustre comment construire un système RAG complet incluant :
- **Ingestion** de documents avec chunking automatique
- **Vectorisation** via sentence-transformers (local)
- **Recherche vectorielle** avec FAISS (en mémoire)
- **Cache intelligent** avec TTL pour optimiser les performances
- **Génération** de réponses (avec adapter LLM mockable)

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Chunking   │───▶│   Embeddings    │
│  (.md, .txt)    │    │   (tokens)   │    │ (sentence-bert) │
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                     │
┌─────────────────┐    ┌──────────────┐    ┌─────────▼─────────┐
│    Response     │◀───│     LLM      │◀───│  Vector Store     │
│   (avec cache)  │    │  (mockable)  │    │    (FAISS)        │
└─────────────────┘    └──────────────┘    └───────────────────┘
                              ▲                      │
                              │            ┌─────────▼─────────┐
                              │            │    Retrieval      │
                              │            │   (k-NN + filter) │
                              │            └─────────▲─────────┘
                              │                      │
                              └──────────────────────┘
                                     Query
```

## 🚀 Démarrage rapide

### Installation

```bash
# Cloner le dépôt
git clone <repo-url>
cd rag_caching

# Installer les dépendances
pip install -r requirements.txt
```

### Lancement de la démo

```bash
# Script de démo complet
python demo.py

# Ou via le script bash
./demo.sh
```

### Tests

```bash
# Lancer tous les tests
pytest tests/

# Tests avec verbosité
pytest -v tests/

# Tests spécifiques
pytest tests/test_rag_flow.py -v
```

## 📁 Structure du projet

```
rag_cashing/
├── README.md                    # Ce fichier
├── requirements.txt             # Dépendances Python
├── config.yaml                  # Configuration globale
├── demo.py                      # Script de démonstration
├── demo.sh                      # Script bash de lancement
│
├── src/                         # Code source principal
│   ├── __init__.py
│   ├── embedder.py              # Interface + sentence-transformers
│   ├── vector_store.py          # Wrapper FAISS avec persistance
│   ├── retriever.py             # Logique de récupération k-NN
│   ├── rag_engine.py            # Orchestration complète + cache
│   ├── llm_adapter.py           # Interface LLM + implémentation mock
│   └── utils.py                 # Utilitaires (chargement, chunking)
│
├── docs/                        # Documentation
│   └── RAG.md                   # Méthodologie et concepts
│
├── samples/                     # Données d'exemple
│   └── docs_to_ingest/          # Documents pour démo
│       ├── introduction_rag.md
│       ├── vectorisation.md
│       └── cache_optimisation.md
│
├── tests/                       # Tests unitaires
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   └── test_rag_flow.py
│
└── archive/                     # Archivage des versions précédentes
    └── ORIG_20250921_225920/
```

## ⚙️ Configuration

Le fichier `config.yaml` permet d'ajuster les paramètres :

```yaml
embedding:
  model_name: "all-MiniLM-L6-v2"  # Modèle sentence-transformers

vector_store:
  dimension: 384                   # Dimension des embeddings

retrieval:
  k: 4                            # Nombre de documents à récupérer

cache:
  ttl_seconds: 3600               # Durée de vie du cache (1h)
  max_size: 1000                  # Taille max du cache

chunking:
  max_tokens: 500                 # Taille max des chunks
  overlap: 50                     # Chevauchement entre chunks
```

## 🔧 Utilisation du code

### Exemple basique

```python
from src.embedder import SentenceTransformersEmbedder
from src.vector_store import VectorStore
from src.llm_adapter import MockLLM
from src.rag_engine import RAGEngine

# Initialisation
embedder = SentenceTransformersEmbedder()
vector_store = VectorStore(dimension=384)
llm_adapter = MockLLM()

# Moteur RAG
rag = RAGEngine(embedder, vector_store, llm_adapter)

# Ingestion de documents
documents = [
    {"text": "Le RAG combine recherche et génération...", "source": "doc1.md"},
    {"text": "FAISS est une bibliothèque de recherche vectorielle...", "source": "doc2.md"}
]
rag.ingest_documents(documents)

# Requête
result = rag.answer("Qu'est-ce que le RAG ?")
print(result['answer'])
```

### Persistance

```python
# Sauvegarde du vector store
vector_store.save("./mon_index")

# Chargement
vector_store.load("./mon_index")
```

### Cache personnalisé

```python
# Désactiver le cache
rag = RAGEngine(embedder, vector_store, llm_adapter, enable_cache=False)

# Cache avec TTL personnalisé
rag = RAGEngine(embedder, vector_store, llm_adapter, cache_ttl=1800)  # 30min
```

## 🧪 Tests et validation

Les tests couvrent :
- ✅ Embedding de textes
- ✅ Stockage et recherche vectorielle
- ✅ Récupération avec filtrage
- ✅ Cache avec TTL
- ✅ Flux RAG complet

## 📊 Performances

La démo inclut des métriques de performance :
- Temps de traitement des requêtes
- Efficacité du cache (hits/misses)
- Statistiques du vector store

## 🔮 Extensions possibles

Ce projet sert de base pour :

1. **LLM réels** : Remplacer MockLLM par OpenAI, Anthropic, etc.
2. **Base vectorielle persistante** : Chroma, Pinecone, Weaviate
3. **Recherche hybride** : Combinaison dense + sparse (BM25)
4. **Métadonnées avancées** : Filtrage par date, auteur, type
5. **Interface web** : Streamlit, FastAPI, Gradio
6. **Monitoring** : Métriques, logs, observabilité

## 🤝 Contribution

Contributions bienvenues ! Veuillez :
1. Forker le projet
2. Créer une branche feature
3. Ajouter des tests pour les nouvelles fonctionnalités
4. Soumettre une pull request

## 📝 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 📚 Ressources

- [Documentation RAG](docs/RAG.md) - Concepts et méthodologie
- [Sentence Transformers](https://www.sbert.net/) - Modèles d'embedding
- [FAISS](https://github.com/facebookresearch/faiss) - Recherche vectorielle
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) - Article original
