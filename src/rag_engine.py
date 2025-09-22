"""
Moteur RAG principal pour orchestrer l'ingestion, la récupération et la génération.
Inclut un système de cache simple pour optimiser les performances.
"""

from typing import List, Dict, Any, Optional
import logging
import time
import hashlib
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .llm_adapter import LLMAdapter

log = logging.getLogger(__name__)


class SimpleCache:
    """Cache simple avec TTL pour les réponses RAG."""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        Initialise le cache.
        
        Args:
            ttl_seconds: Durée de vie des entrées en secondes
            max_size: Taille maximale du cache
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        log.info(f"Cache initialisé: TTL={ttl_seconds}s, max_size={max_size}")
    
    def _get_key(self, query: str, k: int) -> str:
        """Génère une clé de cache basée sur la requête et les paramètres."""
        content = f"{query}_{k}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, query: str, k: int) -> Optional[str]:
        """
        Récupère une réponse du cache si elle existe et est valide.
        
        Args:
            query: Requête originale
            k: Nombre de documents utilisés
            
        Returns:
            Réponse mise en cache ou None
        """
        key = self._get_key(query, k)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        current_time = time.time()
        
        # Vérification de l'expiration
        if current_time - entry['timestamp'] > self.ttl_seconds:
            del self.cache[key]
            log.debug(f"Entrée cache expirée supprimée: {key}")
            return None
        
        log.debug(f"Cache hit pour la clé: {key}")
        return entry['response']
    
    def put(self, query: str, k: int, response: str) -> None:
        """
        Met une réponse en cache.
        
        Args:
            query: Requête originale
            k: Nombre de documents utilisés
            response: Réponse à mettre en cache
        """
        # Nettoyage si le cache est plein
        if len(self.cache) >= self.max_size:
            self._cleanup_expired()
            
            # Si toujours plein, supprimer les entrées les plus anciennes
            if len(self.cache) >= self.max_size:
                oldest_keys = sorted(
                    self.cache.keys(),
                    key=lambda k: self.cache[k]['timestamp']
                )[:len(self.cache) - self.max_size + 1]
                
                for key in oldest_keys:
                    del self.cache[key]
                log.debug(f"Supprimé {len(oldest_keys)} entrées anciennes du cache")
        
        key = self._get_key(query, k)
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }
        log.debug(f"Nouvelle entrée mise en cache: {key}")
    
    def _cleanup_expired(self) -> None:
        """Nettoie les entrées expirées du cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry['timestamp'] > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            log.debug(f"Nettoyé {len(expired_keys)} entrées expirées du cache")
    
    def clear(self) -> None:
        """Vide le cache."""
        self.cache.clear()
        log.info("Cache vidé")
    
    def stats(self) -> Dict[str, int]:
        """Retourne les statistiques du cache."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


class RAGEngine:
    """Moteur RAG principal orchestrant toutes les composantes."""
    
    def __init__(self, embedder: Embedder, vector_store: VectorStore, 
                 llm_adapter: LLMAdapter, k: int = 4, 
                 enable_cache: bool = True, cache_ttl: int = 3600):
        """
        Initialise le moteur RAG.
        
        Args:
            embedder: Instance d'embedder
            vector_store: Store de vecteurs
            llm_adapter: Adapteur LLM
            k: Nombre de documents à récupérer
            enable_cache: Activer le cache
            cache_ttl: TTL du cache en secondes
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_adapter = llm_adapter
        self.retriever = Retriever(embedder, vector_store, k)
        
        # Cache optionnel
        self.enable_cache = enable_cache
        self.cache = SimpleCache(ttl_seconds=cache_ttl) if enable_cache else None
        
        log.info(f"RAGEngine initialisé (k={k}, cache={'activé' if enable_cache else 'désactivé'})")
    
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Ingère des documents dans le système RAG.
        
        Args:
            documents: Liste de documents avec au minimum les clés 'text' et 'source'
        """
        if not documents:
            log.warning("Aucun document à ingérer")
            return
        
        log.info(f"Début de l'ingestion de {len(documents)} documents")
        
        # Validation des documents
        valid_docs = []
        for i, doc in enumerate(documents):
            if 'text' not in doc or not doc['text'].strip():
                log.warning(f"Document {i} ignoré: pas de texte")
                continue
            valid_docs.append(doc)
        
        if not valid_docs:
            log.error("Aucun document valide à ingérer")
            return
        
        # Extraction des textes pour embedding
        texts = [doc['text'] for doc in valid_docs]
        
        # Génération des embeddings
        log.debug("Génération des embeddings...")
        embeddings = self.embedder.embed(texts)
        
        # Préparation des métadonnées
        metadatas = []
        for doc in valid_docs:
            metadata = {
                'text': doc['text'],
                'source': doc.get('source', 'source_inconnue'),
                'chunk_id': doc.get('chunk_id', ''),
                'ingestion_time': time.time()
            }
            # Ajout d'autres métadonnées si présentes
            for key, value in doc.items():
                if key not in ['text', 'source', 'chunk_id']:
                    metadata[key] = value
            metadatas.append(metadata)
        
        # Ajout au vector store
        self.vector_store.add(embeddings, metadatas)
        
        # Invalidation du cache car de nouveaux documents sont disponibles
        if self.cache:
            self.cache.clear()
            log.debug("Cache invalidé après ingestion")
        
        log.info(f"Ingestion terminée: {len(valid_docs)} documents ajoutés")
    
    def answer(self, query: str, k: Optional[int] = None, 
               max_context_length: Optional[int] = None,
               use_cache: Optional[bool] = None) -> Dict[str, Any]:
        """
        Génère une réponse à une question en utilisant RAG.
        
        Args:
            query: Question de l'utilisateur
            k: Nombre de documents à récupérer (optionnel)
            max_context_length: Longueur max du contexte (optionnel)
            use_cache: Utiliser le cache pour cette requête (optionnel)
            
        Returns:
            Dictionnaire avec la réponse et les métadonnées
        """
        if not query.strip():
            return {
                'answer': "Erreur: question vide",
                'query': query,
                'documents': [],
                'cached': False,
                'processing_time': 0
            }
        
        start_time = time.time()
        k = k or self.retriever.k
        use_cache = use_cache if use_cache is not None else self.enable_cache
        
        log.info(f"Traitement de la question: '{query[:50]}...'")
        
        # Vérification du cache
        cached_response = None
        if use_cache and self.cache:
            cached_response = self.cache.get(query, k)
            if cached_response:
                processing_time = time.time() - start_time
                log.info(f"Réponse trouvée en cache (temps: {processing_time:.3f}s)")
                return {
                    'answer': cached_response,
                    'query': query,
                    'documents': [],  # Les documents ne sont pas stockés en cache
                    'cached': True,
                    'processing_time': processing_time
                }
        
        # Récupération des documents pertinents
        log.debug("Récupération des documents...")
        documents = self.retriever.retrieve(query, k=k)
        
        if not documents:
            response = "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question."
            processing_time = time.time() - start_time
            return {
                'answer': response,
                'query': query,
                'documents': documents,
                'cached': False,
                'processing_time': processing_time
            }
        
        # Construction du contexte
        context = self.retriever.get_context_text(documents, max_context_length)
        
        # Composition du prompt
        prompt = self._build_prompt(query, context)
        
        # Génération de la réponse
        log.debug("Génération de la réponse LLM...")
        answer = self.llm_adapter.generate(prompt)
        
        # Mise en cache de la réponse
        if use_cache and self.cache:
            self.cache.put(query, k, answer)
        
        processing_time = time.time() - start_time
        log.info(f"Question traitée en {processing_time:.3f}s")
        
        return {
            'answer': answer,
            'query': query,
            'documents': documents,
            'cached': False,
            'processing_time': processing_time,
            'context_length': len(context)
        }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Construit le prompt pour le LLM.
        
        Args:
            query: Question de l'utilisateur
            context: Contexte récupéré
            
        Returns:
            Prompt formaté
        """
        return f"""Utilisez le contexte suivant pour répondre à la question de manière précise et détaillée.

CONTEXTE:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Répondez uniquement en vous basant sur le contexte fourni
- Si l'information n'est pas dans le contexte, indiquez-le clairement
- Soyez précis et concis
- Citez les sources quand c'est pertinent

RÉPONSE:"""
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du système RAG.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        stats = {
            'vector_store_size': self.vector_store.size,
            'embedder_model': getattr(self.embedder, 'model_name', 'unknown'),
            'retrieval_k': self.retriever.k,
            'cache_enabled': self.enable_cache
        }
        
        if self.cache:
            stats['cache_stats'] = self.cache.stats()
        
        return stats
