"""
Module d'embedding pour la démo RAG.
Fournit une interface abstraite et une implémentation avec sentence-transformers.
"""

from typing import List
import logging
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


class Embedder:
    """Interface abstraite pour les embedders."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Convertit une liste de textes en vecteurs d'embedding.
        
        Args:
            texts: Liste des textes à embedder
            
        Returns:
            Liste des vecteurs d'embedding
        """
        raise NotImplementedError


class SentenceTransformersEmbedder(Embedder):
    """Implémentation avec sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialise l'embedder avec un modèle sentence-transformers.
        
        Args:
            model_name: Nom du modèle à utiliser
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        log.info(f"Embedder chargé : {model_name}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embedde les textes avec sentence-transformers.
        
        Args:
            texts: Textes à embedder
            
        Returns:
            Vecteurs d'embedding
        """
        if not texts:
            return []
        
        log.debug(f"Embedding {len(texts)} textes...")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        """Retourne la dimension des embeddings."""
        return self.model.get_sentence_embedding_dimension()
