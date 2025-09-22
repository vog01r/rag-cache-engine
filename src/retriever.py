"""
Module de récupération pour la démo RAG.
Orchestration de la recherche vectorielle avec filtrage optionnel.
"""

from typing import List, Dict, Any, Optional
import logging
from .embedder import Embedder
from .vector_store import VectorStore

log = logging.getLogger(__name__)


class Retriever:
    """Récupérateur de documents basé sur la similarité vectorielle."""
    
    def __init__(self, embedder: Embedder, vector_store: VectorStore, k: int = 4):
        """
        Initialise le retriever.
        
        Args:
            embedder: Instance d'embedder pour convertir les requêtes
            vector_store: Store de vecteurs pour la recherche
            k: Nombre de documents à récupérer par défaut
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.k = k
        log.info(f"Retriever initialisé avec k={k}")
    
    def retrieve(self, query: str, k: Optional[int] = None, 
                 filter_func: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Récupère les documents les plus pertinents pour une requête.
        
        Args:
            query: Requête textuelle
            k: Nombre de documents à récupérer (optionnel)
            filter_func: Fonction de filtrage des métadonnées (optionnel)
            
        Returns:
            Liste des documents avec leurs métadonnées et scores
        """
        if not query.strip():
            log.warning("Requête vide")
            return []
        
        k = k or self.k
        
        # Embedding de la requête
        log.debug(f"Embedding de la requête: '{query[:50]}...'")
        query_embedding = self.embedder.embed([query])[0]
        
        # Recherche vectorielle
        results = self.vector_store.query(query_embedding, k=k)
        
        # Filtrage optionnel
        if filter_func:
            results = [r for r in results if filter_func(r)]
            log.debug(f"Après filtrage: {len(results)} résultats")
        
        log.info(f"Récupéré {len(results)} documents pour la requête")
        return results
    
    def retrieve_with_source_filter(self, query: str, allowed_sources: List[str], 
                                  k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Récupère des documents en filtrant par source.
        
        Args:
            query: Requête textuelle
            allowed_sources: Liste des sources autorisées
            k: Nombre de documents à récupérer
            
        Returns:
            Documents filtrés par source
        """
        def source_filter(metadata: Dict[str, Any]) -> bool:
            source = metadata.get('source', '')
            return any(allowed_source in source for allowed_source in allowed_sources)
        
        return self.retrieve(query, k=k, filter_func=source_filter)
    
    def get_context_text(self, documents: List[Dict[str, Any]], 
                        max_length: Optional[int] = None) -> str:
        """
        Combine les documents en un texte de contexte pour le LLM.
        
        Args:
            documents: Liste des documents récupérés
            max_length: Longueur maximale du contexte (optionnel)
            
        Returns:
            Texte de contexte formaté
        """
        if not documents:
            return ""
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents):
            text = doc.get('text', '')
            source = doc.get('source', 'Source inconnue')
            
            # Format du contexte
            part = f"[Document {i+1} - {source}]\n{text}\n"
            
            # Vérification de la longueur si spécifiée
            if max_length and total_length + len(part) > max_length:
                if total_length == 0:  # Au moins un document
                    context_parts.append(part[:max_length])
                break
            
            context_parts.append(part)
            total_length += len(part)
        
        context = "\n".join(context_parts)
        log.debug(f"Contexte généré: {len(context)} caractères, {len(context_parts)} documents")
        
        return context
