"""
Adapteur LLM pour la démo RAG.
Interface abstraite et implémentation mock pour démonstration.
"""

from typing import Optional
import logging

log = logging.getLogger(__name__)


class LLMAdapter:
    """Interface abstraite pour les modèles de langage."""
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Génère une réponse à partir d'un prompt.
        
        Args:
            prompt: Prompt d'entrée
            max_tokens: Nombre maximum de tokens (optionnel)
            
        Returns:
            Réponse générée
        """
        raise NotImplementedError


class MockLLM(LLMAdapter):
    """Implémentation mock pour démonstration."""
    
    def __init__(self, response_template: Optional[str] = None):
        """
        Initialise le LLM mock.
        
        Args:
            response_template: Template de réponse (optionnel)
        """
        self.response_template = response_template or (
            "RÉPONSE MOCKÉE basée sur le contexte fourni:\n\n"
            "Contexte analysé: {context_preview}\n\n"
            "Question: {question}\n\n"
            "Cette réponse est générée par un LLM simulé. "
            "Dans une implémentation réelle, vous pourriez utiliser OpenAI, "
            "Anthropic, ou tout autre service LLM."
        )
        log.info("MockLLM initialisé")
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Génère une réponse mock basée sur le prompt.
        
        Args:
            prompt: Prompt complet avec contexte et question
            max_tokens: Ignoré dans cette implémentation mock
            
        Returns:
            Réponse formatée
        """
        # Extraction simple du contexte et de la question
        lines = prompt.split('\n')
        context_preview = ""
        question = ""
        
        # Recherche du contexte et de la question
        for i, line in enumerate(lines):
            if "Question:" in line:
                question = line.replace("Question:", "").strip()
                break
            elif line.strip() and len(context_preview) < 200:
                context_preview += line[:100] + "... "
        
        # Génération de la réponse mock
        response = self.response_template.format(
            context_preview=context_preview.strip() or "Contexte non détecté",
            question=question or "Question non détectée"
        )
        
        # Limitation de longueur si spécifiée
        if max_tokens:
            # Approximation: 1 token ≈ 4 caractères
            max_chars = max_tokens * 4
            if len(response) > max_chars:
                response = response[:max_chars] + "..."
        
        log.debug(f"Réponse mock générée: {len(response)} caractères")
        return response


class OpenAIAdapter(LLMAdapter):
    """Adapteur pour OpenAI (implémentation de base)."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialise l'adapteur OpenAI.
        
        Args:
            api_key: Clé API OpenAI
            model: Modèle à utiliser
        """
        self.api_key = api_key
        self.model = model
        log.info(f"OpenAI adapter configuré avec le modèle {model}")
        
        # Note: Dans une vraie implémentation, vous initialiseriez ici le client OpenAI
        # import openai
        # self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Génère une réponse avec OpenAI.
        
        Args:
            prompt: Prompt d'entrée
            max_tokens: Nombre maximum de tokens
            
        Returns:
            Réponse générée
        """
        # Implémentation exemple (nécessite la vraie clé API et le client OpenAI)
        log.warning("OpenAIAdapter: implémentation exemple - nécessite une vraie clé API")
        
        # Code exemple pour une vraie implémentation:
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=max_tokens
        # )
        # return response.choices[0].message.content
        
        return "ERREUR: OpenAI adapter nécessite une vraie implémentation avec clé API"
