"""
Tests pour le module embedder.
"""

import pytest
import sys
import os

# Ajout du chemin src pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.embedder import Embedder, SentenceTransformersEmbedder


class TestEmbedder:
    """Tests pour la classe abstraite Embedder."""
    
    def test_embedder_interface(self):
        """Test que la classe abstraite lève NotImplementedError."""
        embedder = Embedder()
        
        with pytest.raises(NotImplementedError):
            embedder.embed(["test"])


class TestSentenceTransformersEmbedder:
    """Tests pour SentenceTransformersEmbedder."""
    
    @pytest.fixture
    def embedder(self):
        """Fixture pour créer un embedder de test."""
        return SentenceTransformersEmbedder("all-MiniLM-L6-v2")
    
    def test_initialization(self, embedder):
        """Test l'initialisation de l'embedder."""
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.model is not None
        assert embedder.dimension == 384  # Dimension du modèle all-MiniLM-L6-v2
    
    def test_embed_single_text(self, embedder):
        """Test l'embedding d'un texte simple."""
        texts = ["Ceci est un test"]
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        assert all(isinstance(x, float) for x in embeddings[0])
    
    def test_embed_multiple_texts(self, embedder):
        """Test l'embedding de plusieurs textes."""
        texts = [
            "Premier texte de test",
            "Deuxième texte différent",
            "Troisième texte pour validation"
        ]
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        
        # Vérifier que les embeddings sont différents
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]
    
    def test_embed_empty_list(self, embedder):
        """Test l'embedding d'une liste vide."""
        embeddings = embedder.embed([])
        assert embeddings == []
    
    def test_embed_similar_texts(self, embedder):
        """Test que des textes similaires ont des embeddings proches."""
        texts = [
            "Le chat mange du poisson",
            "Un chat qui mange du poisson"
        ]
        embeddings = embedder.embed(texts)
        
        # Calcul de la similarité cosinus
        import numpy as np
        
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        
        # Normalisation
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Similarité cosinus
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Textes similaires doivent avoir une similarité élevée
        assert similarity > 0.7
    
    def test_embed_different_texts(self, embedder):
        """Test que des textes différents ont des embeddings distants."""
        texts = [
            "Le chat mange du poisson",
            "Les mathématiques quantiques sont complexes"
        ]
        embeddings = embedder.embed(texts)
        
        import numpy as np
        
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Textes différents doivent avoir une similarité plus faible
        assert similarity < 0.5
    
    def test_embed_consistency(self, embedder):
        """Test que le même texte produit toujours le même embedding."""
        text = "Texte de test pour la consistance"
        
        embedding1 = embedder.embed([text])[0]
        embedding2 = embedder.embed([text])[0]
        
        assert embedding1 == embedding2
    
    def test_dimension_property(self, embedder):
        """Test la propriété dimension."""
        assert isinstance(embedder.dimension, int)
        assert embedder.dimension > 0
    
    def test_different_model(self):
        """Test avec un modèle différent (si disponible)."""
        try:
            # Test avec un modèle plus petit
            embedder = SentenceTransformersEmbedder("paraphrase-MiniLM-L3-v2")
            embeddings = embedder.embed(["test"])
            assert len(embeddings[0]) == embedder.dimension
        except:
            # Si le modèle n'est pas disponible, ignorer le test
            pytest.skip("Modèle paraphrase-MiniLM-L3-v2 non disponible")


if __name__ == "__main__":
    pytest.main([__file__])
