"""
Tests pour le module retriever.
"""

import pytest
import sys
import os

# Ajout du chemin src pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.embedder import SentenceTransformersEmbedder
from src.vector_store import VectorStore
from src.retriever import Retriever


class MockEmbedder:
    """Embedder mocké pour les tests."""
    
    def __init__(self, dimension=3):
        self.dimension = dimension
    
    def embed(self, texts):
        """Retourne des embeddings déterministes basés sur le texte."""
        embeddings = []
        for text in texts:
            # Génération d'embedding simple basée sur la longueur et le contenu
            length = len(text)
            char_sum = sum(ord(c) for c in text[:3])  # Somme des 3 premiers caractères
            
            if self.dimension == 3:
                embedding = [length * 0.1, char_sum * 0.01, length + char_sum * 0.001]
            else:
                embedding = [length * 0.1] * self.dimension
            
            embeddings.append(embedding)
        return embeddings


class TestRetriever:
    """Tests pour la classe Retriever."""
    
    @pytest.fixture
    def mock_embedder(self):
        """Fixture pour un embedder mocké."""
        return MockEmbedder(dimension=3)
    
    @pytest.fixture
    def vector_store(self):
        """Fixture pour un vector store."""
        return VectorStore(dimension=3)
    
    @pytest.fixture
    def retriever(self, mock_embedder, vector_store):
        """Fixture pour un retriever."""
        return Retriever(mock_embedder, vector_store, k=2)
    
    @pytest.fixture
    def populated_retriever(self, retriever):
        """Fixture pour un retriever avec des données."""
        documents = [
            {"text": "Les chats aiment le poisson", "source": "animaux.txt"},
            {"text": "Les chiens adorent jouer", "source": "animaux.txt"},
            {"text": "Python est un langage de programmation", "source": "tech.txt"},
            {"text": "JavaScript est utilisé pour le web", "source": "tech.txt"},
            {"text": "Le machine learning révolutionne l'IA", "source": "ia.txt"}
        ]
        
        # Ajout des documents au vector store
        texts = [doc["text"] for doc in documents]
        embeddings = retriever.embedder.embed(texts)
        retriever.vector_store.add(embeddings, documents)
        
        return retriever
    
    def test_initialization(self, mock_embedder, vector_store):
        """Test l'initialisation du retriever."""
        retriever = Retriever(mock_embedder, vector_store, k=5)
        
        assert retriever.embedder is mock_embedder
        assert retriever.vector_store is vector_store
        assert retriever.k == 5
    
    def test_retrieve_empty_query(self, retriever):
        """Test avec une requête vide."""
        results = retriever.retrieve("")
        assert results == []
        
        results = retriever.retrieve("   ")  # Espaces seulement
        assert results == []
    
    def test_retrieve_empty_store(self, retriever):
        """Test de récupération sur un store vide."""
        results = retriever.retrieve("test query")
        assert results == []
    
    def test_retrieve_basic(self, populated_retriever):
        """Test de récupération basique."""
        results = populated_retriever.retrieve("chats poisson")
        
        assert len(results) <= populated_retriever.k
        assert len(results) > 0
        
        # Vérifier la structure des résultats
        for result in results:
            assert "text" in result
            assert "source" in result
            assert "distance" in result
    
    def test_retrieve_with_custom_k(self, populated_retriever):
        """Test de récupération avec k personnalisé."""
        results = populated_retriever.retrieve("programmation", k=1)
        
        assert len(results) == 1
    
    def test_retrieve_with_filter(self, populated_retriever):
        """Test de récupération avec filtrage."""
        def tech_filter(metadata):
            return metadata.get("source") == "tech.txt"
        
        results = populated_retriever.retrieve("programmation", filter_func=tech_filter)
        
        # Tous les résultats doivent venir de tech.txt
        for result in results:
            assert result["source"] == "tech.txt"
    
    def test_retrieve_with_source_filter(self, populated_retriever):
        """Test de récupération avec filtrage par source."""
        results = populated_retriever.retrieve_with_source_filter(
            "animaux", ["animaux.txt"]
        )
        
        # Tous les résultats doivent venir de animaux.txt
        for result in results:
            assert result["source"] == "animaux.txt"
    
    def test_retrieve_with_multiple_sources(self, populated_retriever):
        """Test de récupération avec plusieurs sources autorisées."""
        results = populated_retriever.retrieve_with_source_filter(
            "test", ["tech.txt", "ia.txt"]
        )
        
        # Les résultats doivent venir seulement des sources autorisées
        allowed_sources = {"tech.txt", "ia.txt"}
        for result in results:
            assert result["source"] in allowed_sources
    
    def test_get_context_text_empty(self, retriever):
        """Test de génération de contexte avec une liste vide."""
        context = retriever.get_context_text([])
        assert context == ""
    
    def test_get_context_text_basic(self, populated_retriever):
        """Test de génération de contexte basique."""
        # Récupérer quelques documents
        results = populated_retriever.retrieve("test", k=2)
        
        context = populated_retriever.get_context_text(results)
        
        assert len(context) > 0
        assert "[Document 1" in context
        assert "[Document 2" in context if len(results) > 1 else True
        
        # Vérifier que le texte des documents est inclus
        for i, result in enumerate(results):
            assert result["text"] in context
    
    def test_get_context_text_with_max_length(self, populated_retriever):
        """Test de génération de contexte avec longueur maximale."""
        results = populated_retriever.retrieve("test", k=3)
        
        # Contexte avec limitation de longueur
        context = populated_retriever.get_context_text(results, max_length=100)
        
        assert len(context) <= 100
        assert len(context) > 0  # Devrait inclure au moins un document
    
    def test_context_text_format(self, populated_retriever):
        """Test du format du texte de contexte."""
        results = populated_retriever.retrieve("programmation", k=1)
        context = populated_retriever.get_context_text(results)
        
        # Vérifier le format attendu
        assert "[Document 1 -" in context
        assert results[0]["source"] in context
        assert results[0]["text"] in context
    
    def test_retrieve_relevance_ranking(self, populated_retriever):
        """Test que les résultats sont triés par pertinence."""
        results = populated_retriever.retrieve("Python programmation", k=3)
        
        # Les distances doivent être croissantes (plus proche = distance plus petite)
        distances = [result["distance"] for result in results]
        assert distances == sorted(distances)
    
    def test_filter_func_none_documents(self, populated_retriever):
        """Test avec un filtre qui ne retourne aucun document."""
        def impossible_filter(metadata):
            return metadata.get("source") == "nonexistent.txt"
        
        results = populated_retriever.retrieve("test", filter_func=impossible_filter)
        assert results == []
    
    def test_embedder_consistency(self, populated_retriever):
        """Test que l'embedder est utilisé de façon consistante."""
        # Même requête, mêmes résultats
        query = "test de consistance"
        
        results1 = populated_retriever.retrieve(query)
        results2 = populated_retriever.retrieve(query)
        
        assert len(results1) == len(results2)
        if results1:  # Si des résultats existent
            assert results1[0]["text"] == results2[0]["text"]
            assert abs(results1[0]["distance"] - results2[0]["distance"]) < 1e-6


class TestRetrieverIntegration:
    """Tests d'intégration avec un vrai embedder."""
    
    @pytest.fixture
    def real_retriever(self):
        """Retriever avec un vrai embedder (test d'intégration)."""
        try:
            embedder = SentenceTransformersEmbedder("all-MiniLM-L6-v2")
            vector_store = VectorStore(dimension=384)
            return Retriever(embedder, vector_store, k=3)
        except:
            pytest.skip("Sentence transformers non disponible pour les tests d'intégration")
    
    def test_real_embedder_similarity(self, real_retriever):
        """Test de similarité avec un vrai embedder."""
        documents = [
            {"text": "Les chats sont des animaux domestiques", "source": "animals.txt"},
            {"text": "Python est un langage de programmation", "source": "programming.txt"},
            {"text": "Les félins incluent les chats et les lions", "source": "animals.txt"}
        ]
        
        # Ajout des documents
        texts = [doc["text"] for doc in documents]
        embeddings = real_retriever.embedder.embed(texts)
        real_retriever.vector_store.add(embeddings, documents)
        
        # Requête sur les chats - devrait récupérer les documents sur les animaux
        results = real_retriever.retrieve("chat animal domestique")
        
        assert len(results) > 0
        # Le premier résultat devrait être pertinent pour les chats
        assert any("chat" in result["text"].lower() for result in results[:2])


if __name__ == "__main__":
    pytest.main([__file__])
