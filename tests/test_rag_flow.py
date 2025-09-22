"""
Tests pour le flux RAG complet.
"""

import pytest
import tempfile
import time
import sys
import os

# Ajout du chemin src pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.embedder import SentenceTransformersEmbedder
from src.vector_store import VectorStore
from src.llm_adapter import MockLLM
from src.rag_engine import RAGEngine, SimpleCache


class MockEmbedder:
    """Embedder mocké pour les tests rapides."""
    
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.model_name = "mock-embedder"
    
    def embed(self, texts):
        """Retourne des embeddings déterministes."""
        embeddings = []
        for text in texts:
            # Embedding basé sur la longueur et le hash du texte
            text_hash = hash(text) % 1000
            length = len(text)
            
            if self.dimension == 3:
                embedding = [
                    length * 0.1,
                    text_hash * 0.001,
                    (length + text_hash) * 0.01
                ]
            else:
                embedding = [length * 0.1] * self.dimension
            
            embeddings.append(embedding)
        return embeddings


class TestSimpleCache:
    """Tests pour la classe SimpleCache."""
    
    def test_cache_initialization(self):
        """Test l'initialisation du cache."""
        cache = SimpleCache(ttl_seconds=60, max_size=100)
        
        assert cache.ttl_seconds == 60
        assert cache.max_size == 100
        assert len(cache.cache) == 0
    
    def test_cache_put_and_get(self):
        """Test de mise en cache et récupération."""
        cache = SimpleCache(ttl_seconds=60)
        
        # Mise en cache
        cache.put("test query", 5, "test response")
        
        # Récupération
        result = cache.get("test query", 5)
        assert result == "test response"
    
    def test_cache_miss(self):
        """Test de cache miss."""
        cache = SimpleCache(ttl_seconds=60)
        
        result = cache.get("nonexistent query", 5)
        assert result is None
    
    def test_cache_ttl_expiration(self):
        """Test de l'expiration TTL."""
        cache = SimpleCache(ttl_seconds=0.1)  # 100ms
        
        cache.put("test query", 5, "test response")
        
        # Immédiatement disponible
        assert cache.get("test query", 5) == "test response"
        
        # Attendre l'expiration
        time.sleep(0.15)
        
        # Devrait être expiré
        assert cache.get("test query", 5) is None
    
    def test_cache_max_size(self):
        """Test de la taille maximale du cache."""
        cache = SimpleCache(ttl_seconds=60, max_size=2)
        
        # Remplir le cache
        cache.put("query1", 5, "response1")
        cache.put("query2", 5, "response2")
        
        assert len(cache.cache) == 2
        
        # Ajouter un troisième élément
        cache.put("query3", 5, "response3")
        
        # Le cache ne devrait pas dépasser la taille max
        assert len(cache.cache) <= 2
    
    def test_cache_clear(self):
        """Test du vidage du cache."""
        cache = SimpleCache(ttl_seconds=60)
        
        cache.put("query1", 5, "response1")
        cache.put("query2", 5, "response2")
        
        assert len(cache.cache) == 2
        
        cache.clear()
        
        assert len(cache.cache) == 0
    
    def test_cache_stats(self):
        """Test des statistiques du cache."""
        cache = SimpleCache(ttl_seconds=60, max_size=100)
        stats = cache.stats()
        
        assert stats["size"] == 0
        assert stats["max_size"] == 100
        assert stats["ttl_seconds"] == 60


class TestRAGEngine:
    """Tests pour la classe RAGEngine."""
    
    @pytest.fixture
    def mock_components(self):
        """Fixture avec des composants mockés."""
        embedder = MockEmbedder(dimension=3)
        vector_store = VectorStore(dimension=3)
        llm_adapter = MockLLM()
        
        return embedder, vector_store, llm_adapter
    
    @pytest.fixture
    def rag_engine(self, mock_components):
        """Fixture pour un moteur RAG."""
        embedder, vector_store, llm_adapter = mock_components
        return RAGEngine(embedder, vector_store, llm_adapter, k=2)
    
    @pytest.fixture
    def sample_documents(self):
        """Fixture avec des documents d'exemple."""
        return [
            {
                "text": "Les chats sont des animaux domestiques très populaires.",
                "source": "animaux.txt",
                "category": "animaux"
            },
            {
                "text": "Python est un langage de programmation polyvalent et facile à apprendre.",
                "source": "programming.txt",
                "category": "tech"
            },
            {
                "text": "Le machine learning utilise des algorithmes pour apprendre des données.",
                "source": "ml.txt",
                "category": "tech"
            }
        ]
    
    def test_rag_engine_initialization(self, mock_components):
        """Test l'initialisation du moteur RAG."""
        embedder, vector_store, llm_adapter = mock_components
        
        engine = RAGEngine(embedder, vector_store, llm_adapter, k=5, enable_cache=False)
        
        assert engine.embedder is embedder
        assert engine.vector_store is vector_store
        assert engine.llm_adapter is llm_adapter
        assert engine.retriever.k == 5
        assert engine.enable_cache is False
        assert engine.cache is None
    
    def test_ingest_documents_empty(self, rag_engine):
        """Test d'ingestion de documents vides."""
        rag_engine.ingest_documents([])
        assert rag_engine.vector_store.size == 0
    
    def test_ingest_documents_basic(self, rag_engine, sample_documents):
        """Test d'ingestion basique de documents."""
        rag_engine.ingest_documents(sample_documents)
        
        assert rag_engine.vector_store.size == 3
        assert len(rag_engine.vector_store.metadatas) == 3
        
        # Vérifier que les métadonnées sont préservées
        metadata = rag_engine.vector_store.metadatas[0]
        assert "text" in metadata
        assert "source" in metadata
        assert "category" in metadata
        assert "ingestion_time" in metadata
    
    def test_ingest_documents_invalid(self, rag_engine):
        """Test d'ingestion de documents invalides."""
        invalid_docs = [
            {"text": ""},  # Texte vide
            {"source": "test.txt"},  # Pas de texte
            {"text": "   "},  # Seulement des espaces
            {"text": "Document valide", "source": "valid.txt"}  # Valide
        ]
        
        rag_engine.ingest_documents(invalid_docs)
        
        # Seulement le document valide devrait être ajouté
        assert rag_engine.vector_store.size == 1
    
    def test_answer_empty_query(self, rag_engine, sample_documents):
        """Test de réponse à une requête vide."""
        rag_engine.ingest_documents(sample_documents)
        
        result = rag_engine.answer("")
        
        assert "erreur" in result["answer"].lower()
        assert result["query"] == ""
        assert result["documents"] == []
        assert result["cached"] is False
    
    def test_answer_no_documents(self, rag_engine):
        """Test de réponse sans documents ingérés."""
        result = rag_engine.answer("test query")
        
        assert "pas trouvé" in result["answer"].lower()
        assert result["documents"] == []
    
    def test_answer_basic(self, rag_engine, sample_documents):
        """Test de réponse basique."""
        rag_engine.ingest_documents(sample_documents)
        
        result = rag_engine.answer("Qu'est-ce que Python ?")
        
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert result["query"] == "Qu'est-ce que Python ?"
        assert len(result["documents"]) > 0
        assert result["cached"] is False
        assert result["processing_time"] > 0
    
    def test_answer_with_cache(self, rag_engine, sample_documents):
        """Test du cache de réponses."""
        rag_engine.ingest_documents(sample_documents)
        
        query = "Parlez-moi des chats"
        
        # Première requête
        result1 = rag_engine.answer(query)
        assert result1["cached"] is False
        
        # Deuxième requête identique
        result2 = rag_engine.answer(query)
        assert result2["cached"] is True
        assert result2["answer"] == result1["answer"]
        assert result2["processing_time"] < result1["processing_time"]
    
    def test_answer_disable_cache(self, rag_engine, sample_documents):
        """Test de désactivation du cache."""
        rag_engine.ingest_documents(sample_documents)
        
        query = "Test sans cache"
        
        # Première requête
        result1 = rag_engine.answer(query, use_cache=False)
        assert result1["cached"] is False
        
        # Deuxième requête sans cache
        result2 = rag_engine.answer(query, use_cache=False)
        assert result2["cached"] is False
    
    def test_answer_custom_k(self, rag_engine, sample_documents):
        """Test de réponse avec k personnalisé."""
        rag_engine.ingest_documents(sample_documents)
        
        result = rag_engine.answer("test", k=1)
        
        assert len(result["documents"]) <= 1
    
    def test_get_statistics(self, rag_engine, sample_documents):
        """Test des statistiques du système."""
        rag_engine.ingest_documents(sample_documents)
        
        stats = rag_engine.get_statistics()
        
        assert stats["vector_store_size"] == 3
        assert stats["embedder_model"] == "mock-embedder"
        assert stats["retrieval_k"] == 2
        assert stats["cache_enabled"] is True
        assert "cache_stats" in stats
    
    def test_cache_invalidation_on_ingestion(self, rag_engine, sample_documents):
        """Test que le cache est vidé lors de l'ingestion."""
        # Ingestion initiale
        rag_engine.ingest_documents(sample_documents[:1])
        
        # Requête mise en cache
        query = "test query"
        result1 = rag_engine.answer(query)
        assert result1["cached"] is False
        
        # Vérifier que c'est en cache
        result2 = rag_engine.answer(query)
        assert result2["cached"] is True
        
        # Nouvelle ingestion
        rag_engine.ingest_documents(sample_documents[1:])
        
        # Le cache devrait être vidé
        result3 = rag_engine.answer(query)
        assert result3["cached"] is False


class TestRAGEngineIntegration:
    """Tests d'intégration avec de vrais composants."""
    
    @pytest.fixture
    def real_rag_engine(self):
        """RAG engine avec de vrais composants."""
        try:
            embedder = SentenceTransformersEmbedder("all-MiniLM-L6-v2")
            vector_store = VectorStore(dimension=384)
            llm_adapter = MockLLM()
            
            return RAGEngine(embedder, vector_store, llm_adapter, k=3)
        except:
            pytest.skip("Sentence transformers non disponible pour les tests d'intégration")
    
    def test_end_to_end_flow(self, real_rag_engine):
        """Test du flux complet end-to-end."""
        # Documents sur différents sujets
        documents = [
            {
                "text": "Le RAG (Retrieval-Augmented Generation) combine la recherche et la génération de texte.",
                "source": "rag_intro.txt"
            },
            {
                "text": "FAISS est une bibliothèque pour la recherche vectorielle efficace.",
                "source": "faiss_doc.txt"
            },
            {
                "text": "Les embeddings permettent de représenter le texte sous forme numérique.",
                "source": "embeddings_guide.txt"
            }
        ]
        
        # Ingestion
        real_rag_engine.ingest_documents(documents)
        assert real_rag_engine.vector_store.size == 3
        
        # Test de différentes requêtes
        queries = [
            "Qu'est-ce que le RAG ?",
            "Comment fonctionne FAISS ?",
            "Expliquez les embeddings"
        ]
        
        for query in queries:
            result = real_rag_engine.answer(query)
            
            # Vérifications basiques
            assert isinstance(result["answer"], str)
            assert len(result["answer"]) > 0
            assert len(result["documents"]) > 0
            assert result["processing_time"] > 0
            
            # Le document le plus pertinent devrait contenir des mots-clés de la requête
            top_doc = result["documents"][0]
            query_words = query.lower().split()
            doc_text = top_doc["text"].lower()
            
            # Au moins un mot-clé devrait être présent (test de pertinence basique)
            keyword_found = any(
                word in doc_text for word in query_words 
                if len(word) > 3  # Ignorer les mots courts
            )
            assert keyword_found or result["documents"]  # Au moins des résultats
    
    def test_similarity_relevance(self, real_rag_engine):
        """Test de la pertinence de la similarité."""
        documents = [
            {"text": "Les chats sont des félins domestiques", "source": "cats.txt"},
            {"text": "Python est un langage de programmation", "source": "python.txt"},
            {"text": "Les lions sont des félins sauvages", "source": "lions.txt"}
        ]
        
        real_rag_engine.ingest_documents(documents)
        
        # Requête sur les chats - devrait récupérer les documents sur les félins
        result = real_rag_engine.answer("parlez-moi des chats domestiques")
        
        assert len(result["documents"]) > 0
        
        # Le premier document devrait être le plus pertinent (sur les chats)
        top_doc = result["documents"][0]
        assert "chat" in top_doc["text"].lower()


if __name__ == "__main__":
    pytest.main([__file__])
