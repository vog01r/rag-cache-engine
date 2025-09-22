#!/bin/bash

# Script de démonstration RAG Caching
# Ce script configure l'environnement et lance la démonstration

echo "=============================================="
echo "  DÉMONSTRATION RAG CACHING"
echo "=============================================="

# Vérification de Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trouvé. Veuillez l'installer."
    exit 1
fi

echo "✓ Python 3 détecté: $(python3 --version)"

# Installation des dépendances
echo ""
echo "📦 Installation des dépendances..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✓ Dépendances installées avec succès"
    else
        echo "❌ Erreur lors de l'installation des dépendances"
        exit 1
    fi
else
    echo "❌ Fichier requirements.txt non trouvé"
    exit 1
fi

# Lancement de la démonstration
echo ""
echo "🚀 Lancement de la démonstration..."
echo ""

python3 demo.py

echo ""
echo "✨ Démonstration terminée !"
echo ""
echo "💡 Commandes utiles:"
echo "   - Tests: pytest tests/"
echo "   - Documentation: cat docs/RAG.md"
echo "   - Nettoyage: rm -rf demo_vector_store __pycache__ src/__pycache__"
