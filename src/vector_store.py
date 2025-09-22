"""
Store de vecteurs utilisant FAISS pour la démo RAG.
Implémentation simple en mémoire avec possibilité de sauvegarde.
"""

from typing import List, Dict, Any, Optional
import faiss
import numpy as np
import pickle
import os
import logging

log = logging.getLogger(__name__)


class VectorStore:
    """Store de vecteurs basé sur FAISS."""
    
    def __init__(self, dimension: int):
        """
        Initialise le store de vecteurs.
        
        Args:
            dimension: Dimension des vecteurs d'embedding
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadatas: List[Dict[str, Any]] = []
        log.info(f"VectorStore initialisé avec dimension {dimension}")
    
    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        """
        Ajoute des vecteurs et leurs métadonnées au store.
        
        Args:
            vectors: Liste des vecteurs d'embedding
            metadatas: Métadonnées associées à chaque vecteur
        """
        if len(vectors) != len(metadatas):
            raise ValueError("Le nombre de vecteurs doit correspondre au nombre de métadonnées")
        
        if not vectors:
            return
        
        # Conversion en array numpy pour FAISS
        vectors_array = np.array(vectors).astype('float32')
        
        # Vérification de la dimension
        if vectors_array.shape[1] != self.dimension:
            raise ValueError(f"Dimension des vecteurs ({vectors_array.shape[1]}) != dimension attendue ({self.dimension})")
        
        # Ajout à l'index FAISS
        self.index.add(vectors_array)
        self.metadatas.extend(metadatas)
        
        log.info(f"Ajouté {len(vectors)} vecteurs au store (total: {self.index.ntotal})")
    
    def query(self, vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Recherche les k vecteurs les plus similaires.
        
        Args:
            vector: Vecteur de requête
            k: Nombre de résultats à retourner
            
        Returns:
            Liste des métadonnées des documents les plus similaires
        """
        if self.index.ntotal == 0:
            log.warning("VectorStore vide, aucun résultat")
            return []
        
        # Conversion et recherche
        query_vector = np.array([vector]).astype('float32')
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        # Extraction des métadonnées
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # FAISS retourne -1 pour les slots vides
                metadata = self.metadatas[idx].copy()
                metadata['distance'] = float(distances[0][i])
                results.append(metadata)
        
        log.debug(f"Trouvé {len(results)} résultats pour la requête")
        return results
    
    def save(self, directory_path: str) -> None:
        """
        Sauvegarde l'index et les métadonnées sur disque.
        
        Args:
            directory_path: Chemin du dossier de sauvegarde
        """
        os.makedirs(directory_path, exist_ok=True)
        
        # Sauvegarde de l'index FAISS
        index_path = os.path.join(directory_path, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Sauvegarde des métadonnées
        metadata_path = os.path.join(directory_path, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadatas, f)
        
        log.info(f"VectorStore sauvegardé dans {directory_path}")
    
    def load(self, directory_path: str) -> None:
        """
        Charge l'index et les métadonnées depuis le disque.
        
        Args:
            directory_path: Chemin du dossier de sauvegarde
        """
        index_path = os.path.join(directory_path, "index.faiss")
        metadata_path = os.path.join(directory_path, "metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Fichiers de sauvegarde introuvables dans {directory_path}")
        
        # Chargement de l'index FAISS
        self.index = faiss.read_index(index_path)
        
        # Chargement des métadonnées
        with open(metadata_path, "rb") as f:
            self.metadatas = pickle.load(f)
        
        log.info(f"VectorStore chargé depuis {directory_path} ({self.index.ntotal} vecteurs)")
    
    @property
    def size(self) -> int:
        """Retourne le nombre de vecteurs dans le store."""
        return self.index.ntotal
