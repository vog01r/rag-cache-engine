"""
Utilitaires pour la démo RAG: chargement de fichiers, chunking, configuration du logging.
"""

import os
import re
import yaml
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    """
    Configure le logging pour l'application.
    
    Args:
        level: Niveau de logging (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Configuration chargée
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration chargée depuis {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Fichier de config {config_path} non trouvé, utilisation des valeurs par défaut")
        return get_default_config()
    except yaml.YAMLError as e:
        logging.error(f"Erreur lors du parsing du fichier de config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Retourne la configuration par défaut.
    
    Returns:
        Configuration par défaut
    """
    return {
        'embedding': {
            'model_name': 'all-MiniLM-L6-v2'
        },
        'vector_store': {
            'dimension': 384
        },
        'retrieval': {
            'k': 4
        },
        'cache': {
            'ttl_seconds': 3600,
            'max_size': 1000
        },
        'chunking': {
            'max_tokens': 500,
            'overlap': 50
        }
    }


def load_documents_from_directory(directory_path: str, 
                                supported_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Charge tous les documents depuis un répertoire.
    
    Args:
        directory_path: Chemin vers le répertoire
        supported_extensions: Extensions de fichiers supportées
        
    Returns:
        Liste des documents chargés
    """
    if supported_extensions is None:
        supported_extensions = ['.txt', '.md', '.mdx']
    
    directory = Path(directory_path)
    if not directory.exists():
        logging.error(f"Répertoire {directory_path} non trouvé")
        return []
    
    documents = []
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                doc = load_document(str(file_path))
                if doc:
                    documents.append(doc)
            except Exception as e:
                logging.error(f"Erreur lors du chargement de {file_path}: {e}")
    
    logging.info(f"Chargé {len(documents)} documents depuis {directory_path}")
    return documents


def load_document(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Charge un document depuis un fichier.
    
    Args:
        file_path: Chemin vers le fichier
        
    Returns:
        Document chargé ou None en cas d'erreur
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            logging.warning(f"Fichier vide: {file_path}")
            return None
        
        return {
            'text': content,
            'source': file_path,
            'filename': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1]
        }
    
    except UnicodeDecodeError:
        logging.error(f"Erreur d'encodage pour {file_path}")
        return None
    except Exception as e:
        logging.error(f"Erreur lors du chargement de {file_path}: {e}")
        return None


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """
    Divise un texte en chunks avec overlap.
    
    Args:
        text: Texte à diviser
        max_tokens: Taille maximale approximative d'un chunk (en mots)
        overlap: Nombre de mots de recouvrement entre chunks
        
    Returns:
        Liste des chunks
    """
    if not text.strip():
        return []
    
    # Nettoyage basique du texte
    text = clean_text(text)
    
    # Division en mots (approximation simple)
    words = text.split()
    
    if len(words) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_words = words[start:end]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
        
        # Avancer avec overlap
        if end >= len(words):
            break
        start = end - overlap
        
        # Éviter les boucles infinies
        if start <= 0:
            start = end
    
    logging.debug(f"Texte divisé en {len(chunks)} chunks")
    return chunks


def chunk_documents(documents: List[Dict[str, Any]], 
                   max_tokens: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Divise une liste de documents en chunks.
    
    Args:
        documents: Liste des documents à diviser
        max_tokens: Taille maximale d'un chunk
        overlap: Recouvrement entre chunks
        
    Returns:
        Liste des documents chunkés
    """
    chunked_docs = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text.strip():
            continue
        
        chunks = chunk_text(text, max_tokens, overlap)
        
        for i, chunk in enumerate(chunks):
            chunked_doc = doc.copy()
            chunked_doc['text'] = chunk
            chunked_doc['chunk_id'] = f"{doc.get('source', 'unknown')}_{i}"
            chunked_doc['chunk_index'] = i
            chunked_doc['total_chunks'] = len(chunks)
            chunked_docs.append(chunked_doc)
    
    logging.info(f"Documents divisés: {len(documents)} -> {len(chunked_docs)} chunks")
    return chunked_docs


def clean_text(text: str) -> str:
    """
    Nettoie un texte en supprimant les caractères indésirables.
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte nettoyé
    """
    # Suppression des caractères de contrôle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    
    # Normalisation des espaces
    text = re.sub(r'\s+', ' ', text)
    
    # Suppression des espaces en début/fin
    text = text.strip()
    
    return text


def format_file_size(size_bytes: int) -> str:
    """
    Formate une taille de fichier en unités lisibles.
    
    Args:
        size_bytes: Taille en bytes
        
    Returns:
        Taille formatée
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def ensure_directory(directory_path: str) -> None:
    """
    S'assure qu'un répertoire existe, le crée si nécessaire.
    
    Args:
        directory_path: Chemin du répertoire
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)
