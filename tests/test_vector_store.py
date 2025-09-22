"""
Tests pour le module vector_store.
"""

import pytest
import tempfile
import os
import shutil
import sys

# Ajout du chemin src pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.vector_store import VectorStore


class TestVectorStore:
    """Tests pour la classe VectorStore."""
    
    @pytest.fixture
    def vector_store(self):
        """Fixture pour créer un vector store de test."""
        return VectorStore(dimension=3)  # Dimension réduite pour les tests
    
    @pytest.fixture
    def sample_vectors(self):
        """Fixture avec des vecteurs d'exemple."""
        return [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [1.1, 2.1, 3.1]  # Proche du premier
        ]
    
    @pytest.fixture
    def sample_metadatas(self):
        """Fixture avec des métadonnées d'exemple."""
        return [
            {"text": "Premier document", "source": "doc1.txt", "id": 1},
            {"text": "Deuxième document", "source": "doc2.txt", "id": 2},
            {"text": "Troisième document", "source": "doc3.txt", "id": 3},
            {"text": "Quatrième document", "source": "doc4.txt", "id": 4}
        ]
    
    def test_initialization(self, vector_store):
        """Test l'initialisation du vector store."""
        assert vector_store.dimension == 3
        assert vector_store.size == 0
        assert vector_store.metadatas == []
    
    def test_add_vectors(self, vector_store, sample_vectors, sample_metadatas):
        """Test l'ajout de vecteurs."""
        vector_store.add(sample_vectors, sample_metadatas)
        
        assert vector_store.size == 4
        assert len(vector_store.metadatas) == 4
        assert vector_store.metadatas[0]["text"] == "Premier document"
    
    def test_add_empty_vectors(self, vector_store):
        """Test l'ajout de vecteurs vides."""
        vector_store.add([], [])
        assert vector_store.size == 0
    
    def test_add_mismatched_lengths(self, vector_store):
        """Test l'erreur avec des longueurs non correspondantes."""
        vectors = [[1.0, 2.0, 3.0]]
        metadatas = [{"text": "doc1"}, {"text": "doc2"}]  # Trop de métadonnées
        
        with pytest.raises(ValueError, match="Le nombre de vecteurs doit correspondre"):
            vector_store.add(vectors, metadatas)
    
    def test_add_wrong_dimension(self, vector_store):
        """Test l'erreur avec une mauvaise dimension."""
        wrong_vectors = [[1.0, 2.0]]  # Dimension 2 au lieu de 3
        metadatas = [{"text": "doc"}]
        
        with pytest.raises(ValueError, match="Dimension des vecteurs"):
            vector_store.add(wrong_vectors, metadatas)
    
    def test_query_empty_store(self, vector_store):
        """Test de requête sur un store vide."""
        query_vector = [1.0, 2.0, 3.0]
        results = vector_store.query(query_vector, k=5)
        
        assert results == []
    
    def test_query_similar_vectors(self, vector_store, sample_vectors, sample_metadatas):
        """Test de requête avec vecteurs similaires."""
        vector_store.add(sample_vectors, sample_metadatas)
        
        # Requête avec un vecteur proche du premier
        query_vector = [1.05, 2.05, 3.05]
        results = vector_store.query(query_vector, k=2)
        
        assert len(results) == 2
        
        # Le premier résultat devrait être le plus proche
        assert results[0]["text"] == "Premier document"
        assert "distance" in results[0]
        assert results[0]["distance"] < results[1]["distance"]
    
    def test_query_k_larger_than_store(self, vector_store, sample_vectors, sample_metadatas):
        """Test de requête avec k plus grand que le nombre de vecteurs."""
        vector_store.add(sample_vectors[:2], sample_metadatas[:2])  # Seulement 2 vecteurs
        
        query_vector = [1.0, 2.0, 3.0]
        results = vector_store.query(query_vector, k=5)  # Demander 5 résultats
        
        assert len(results) == 2  # Ne peut retourner que ce qui existe
    
    def test_distance_calculation(self, vector_store, sample_vectors, sample_metadatas):
        """Test que les distances sont calculées correctement."""
        vector_store.add(sample_vectors, sample_metadatas)
        
        # Requête avec le vecteur exact du premier élément
        query_vector = sample_vectors[0]  # [1.0, 2.0, 3.0]
        results = vector_store.query(query_vector, k=1)
        
        # La distance devrait être très proche de 0
        assert results[0]["distance"] < 1e-6
    
    def test_metadata_preservation(self, vector_store, sample_vectors, sample_metadatas):
        """Test que les métadonnées sont préservées."""
        vector_store.add(sample_vectors, sample_metadatas)
        
        query_vector = [1.0, 2.0, 3.0]
        results = vector_store.query(query_vector, k=1)
        
        result = results[0]
        assert "text" in result
        assert "source" in result
        assert "id" in result
        assert "distance" in result  # Ajoutée par la requête
    
    def test_save_and_load(self, vector_store, sample_vectors, sample_metadatas):
        """Test de sauvegarde et chargement."""
        # Ajouter des données
        vector_store.add(sample_vectors, sample_metadatas)
        
        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Sauvegarder
            vector_store.save(temp_dir)
            
            # Vérifier que les fichiers existent
            assert os.path.exists(os.path.join(temp_dir, "index.faiss"))
            assert os.path.exists(os.path.join(temp_dir, "metadata.pkl"))
            
            # Créer un nouveau vector store et charger
            new_store = VectorStore(dimension=3)
            new_store.load(temp_dir)
            
            # Vérifier que les données sont identiques
            assert new_store.size == vector_store.size
            assert len(new_store.metadatas) == len(vector_store.metadatas)
            assert new_store.metadatas[0]["text"] == vector_store.metadatas[0]["text"]
            
            # Test de requête sur le store chargé
            query_vector = [1.0, 2.0, 3.0]
            original_results = vector_store.query(query_vector, k=2)
            loaded_results = new_store.query(query_vector, k=2)
            
            assert len(original_results) == len(loaded_results)
            assert original_results[0]["text"] == loaded_results[0]["text"]
    
    def test_load_nonexistent_files(self):
        """Test de chargement de fichiers inexistants."""
        vector_store = VectorStore(dimension=3)
        
        with pytest.raises(FileNotFoundError):
            vector_store.load("/path/that/does/not/exist")
    
    def test_save_creates_directory(self, vector_store, sample_vectors, sample_metadatas):
        """Test que la sauvegarde crée le répertoire s'il n'existe pas."""
        vector_store.add(sample_vectors, sample_metadatas)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "new_directory")
            
            # Le répertoire n'existe pas encore
            assert not os.path.exists(save_path)
            
            # Sauvegarder (devrait créer le répertoire)
            vector_store.save(save_path)
            
            # Vérifier que le répertoire et les fichiers existent
            assert os.path.exists(save_path)
            assert os.path.exists(os.path.join(save_path, "index.faiss"))
            assert os.path.exists(os.path.join(save_path, "metadata.pkl"))
    
    def test_size_property(self, vector_store, sample_vectors, sample_metadatas):
        """Test de la propriété size."""
        assert vector_store.size == 0
        
        vector_store.add(sample_vectors[:2], sample_metadatas[:2])
        assert vector_store.size == 2
        
        vector_store.add(sample_vectors[2:], sample_metadatas[2:])
        assert vector_store.size == 4
    
    def test_incremental_addition(self, vector_store, sample_vectors, sample_metadatas):
        """Test d'ajouts incrémentaux."""
        # Premier ajout
        vector_store.add(sample_vectors[:2], sample_metadatas[:2])
        assert vector_store.size == 2
        
        # Deuxième ajout
        vector_store.add(sample_vectors[2:], sample_metadatas[2:])
        assert vector_store.size == 4
        
        # Test de requête sur tous les vecteurs
        query_vector = [1.0, 2.0, 3.0]
        results = vector_store.query(query_vector, k=4)
        assert len(results) == 4


if __name__ == "__main__":
    pytest.main([__file__])
