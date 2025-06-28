"""LLM-based text correction module."""

import os
import time
from typing import Optional, Dict, Any, List
import logging
from abc import ABC, abstractmethod

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import requests
except ImportError:
    requests = None

from ..utils.config import get_config
from ..utils.logging_config import ErrorLogger, PerformanceLogger

logger = logging.getLogger(__name__)
error_logger = ErrorLogger()
performance_logger = PerformanceLogger("llm")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def correct_text(self, text: str, context: str = "", **kwargs) -> str:
        """Correct OCR text using LLM.
        
        Args:
            text: OCR text to correct
            context: Additional context for correction
            **kwargs: Provider-specific parameters
            
        Returns:
            Corrected text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for text correction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider.
        
        Args:
            config: OpenAI configuration
        """
        self.config = config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        if not openai:
            logger.warning("OpenAI library not available")
            return
        
        api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OpenAI API key not provided")
            return
        
        try:
            self.client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        except Exception as e:
            error_logger.log_llm_error(e, "OpenAI", 0, "initialization")
    
    def correct_text(self, text: str, context: str = "", **kwargs) -> str:
        """Correct text using OpenAI GPT.
        
        Args:
            text: OCR text to correct
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Corrected text
        """
        if not self.is_available():
            return text
        
        try:
            start_time = time.time()
            
            # Build prompt
            prompt = self._build_correction_prompt(text, context, **kwargs)
            
            # API parameters
            model = kwargs.get('model', self.config.get('model', 'gpt-3.5-turbo'))
            max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 1000))
            temperature = kwargs.get('temperature', self.config.get('temperature', 0.1))
            
            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at correcting OCR errors in English text. Return only the corrected text without explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            corrected_text = response.choices[0].message.content.strip()
            
            # Log performance
            processing_time = time.time() - start_time
            tokens_used = response.usage.total_tokens if response.usage else 0
            performance_logger.log_llm_performance("OpenAI", model, processing_time, tokens_used)
            
            logger.debug(f"OpenAI correction completed in {processing_time:.2f}s")
            return corrected_text
            
        except Exception as e:
            error_logger.log_llm_error(e, "OpenAI", len(text), "text correction")
            return text
    
    def _build_correction_prompt(self, text: str, context: str = "", **kwargs) -> str:
        """Build correction prompt for OpenAI."""
        prompt_parts = [
            "Please correct the OCR errors in the following English text.",
            "Preserve the original meaning and structure.",
            "Fix spelling mistakes, character recognition errors, and formatting issues.",
            "Return only the corrected text."
        ]
        
        if context:
            prompt_parts.append(f"\nContext: {context}")
        
        prompt_parts.extend([
            f"\nOCR Text to correct:",
            f'"{text}"',
            "\nCorrected text:"
        ])
        
        return "\n".join(prompt_parts)
    
    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        return self.client is not None


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider for text correction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Anthropic provider.
        
        Args:
            config: Anthropic configuration
        """
        self.config = config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        if not anthropic:
            logger.warning("Anthropic library not available")
            return
        
        api_key = self.config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.warning("Anthropic API key not provided")
            return
        
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info("Anthropic client initialized")
        except Exception as e:
            error_logger.log_llm_error(e, "Anthropic", 0, "initialization")
    
    def correct_text(self, text: str, context: str = "", **kwargs) -> str:
        """Correct text using Anthropic Claude.
        
        Args:
            text: OCR text to correct
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Corrected text
        """
        if not self.is_available():
            return text
        
        try:
            start_time = time.time()
            
            # Build prompt
            prompt = self._build_correction_prompt(text, context, **kwargs)
            
            # API parameters
            model = kwargs.get('model', self.config.get('model', 'claude-3-haiku-20240307'))
            max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 1000))
            
            # Make API call
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            corrected_text = response.content[0].text.strip()
            
            # Log performance
            processing_time = time.time() - start_time
            tokens_used = response.usage.input_tokens + response.usage.output_tokens if response.usage else 0
            performance_logger.log_llm_performance("Anthropic", model, processing_time, tokens_used)
            
            logger.debug(f"Anthropic correction completed in {processing_time:.2f}s")
            return corrected_text
            
        except Exception as e:
            error_logger.log_llm_error(e, "Anthropic", len(text), "text correction")
            return text
    
    def _build_correction_prompt(self, text: str, context: str = "", **kwargs) -> str:
        """Build correction prompt for Anthropic."""
        prompt_parts = [
            "I need you to correct OCR errors in English text.",
            "Please fix spelling mistakes, character recognition errors, and formatting issues.",
            "Preserve the original meaning and structure.",
            "Return only the corrected text without any explanations or additional commentary."
        ]
        
        if context:
            prompt_parts.append(f"\nContext: {context}")
        
        prompt_parts.extend([
            f"\nText to correct:",
            f'"{text}"'
        ])
        
        return "\n".join(prompt_parts)
    
    def is_available(self) -> bool:
        """Check if Anthropic provider is available."""
        return self.client is not None


class LocalLLMProvider(LLMProvider):
    """Local LLM provider (e.g., Ollama) for text correction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize local LLM provider.
        
        Args:
            config: Local LLM configuration
        """
        self.config = config
        self.endpoint = config.get('endpoint', 'http://localhost:11434')
        self.model = config.get('model', 'llama2')
    
    def correct_text(self, text: str, context: str = "", **kwargs) -> str:
        """Correct text using local LLM.
        
        Args:
            text: OCR text to correct
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Corrected text
        """
        if not self.is_available():
            return text
        
        try:
            start_time = time.time()
            
            # Build prompt
            prompt = self._build_correction_prompt(text, context, **kwargs)
            
            # Make request to local LLM
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                corrected_text = result.get('response', text).strip()
                
                # Log performance
                processing_time = time.time() - start_time
                performance_logger.log_llm_performance("Local", self.model, processing_time, 0)
                
                logger.debug(f"Local LLM correction completed in {processing_time:.2f}s")
                return corrected_text
            else:
                logger.warning(f"Local LLM request failed: {response.status_code}")
                return text
                
        except Exception as e:
            error_logger.log_llm_error(e, "Local", len(text), "text correction")
            return text
    
    def _build_correction_prompt(self, text: str, context: str = "", **kwargs) -> str:
        """Build correction prompt for local LLM."""
        prompt_parts = [
            "Correct the OCR errors in this English text.",
            "Fix spelling and character recognition mistakes.",
            "Keep the original meaning and structure.",
            "Return only the corrected text."
        ]
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        prompt_parts.extend([
            f"Text: {text}",
            "Corrected:"
        ])
        
        return "\n".join(prompt_parts)
    
    def is_available(self) -> bool:
        """Check if local LLM is available."""
        if not requests:
            return False
        
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


class LLMCorrector:
    """Main LLM correction manager."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM corrector.
        
        Args:
            config: LLM configuration. If None, uses global config.
        """
        self.config = config or get_config().get_llm_config()
        self.provider = None
        self._initialize_provider()
    
    def _initialize_provider(self) -> None:
        """Initialize the configured LLM provider."""
        provider_name = self.config.get('provider', 'disabled')
        
        if provider_name == 'disabled':
            logger.info("LLM correction disabled")
            return
        
        try:
            if provider_name == 'openai':
                self.provider = OpenAIProvider(self.config.get('openai', {}))
            elif provider_name == 'anthropic':
                self.provider = AnthropicProvider(self.config.get('anthropic', {}))
            elif provider_name == 'local':
                self.provider = LocalLLMProvider(self.config.get('local', {}))
            else:
                logger.warning(f"Unknown LLM provider: {provider_name}")
                return
            
            if self.provider and self.provider.is_available():
                logger.info(f"LLM provider '{provider_name}' initialized successfully")
            else:
                logger.warning(f"LLM provider '{provider_name}' not available")
                self.provider = None
                
        except Exception as e:
            error_logger.log_llm_error(e, provider_name, 0, "provider initialization")
            self.provider = None
    
    def correct_text(self, text: str, confidence: float = 1.0, context: str = "") -> str:
        """Correct OCR text using LLM.
        
        Args:
            text: OCR text to correct
            confidence: OCR confidence score
            context: Additional context for correction
            
        Returns:
            Corrected text or original text if correction not available/needed
        """
        # Check if correction is enabled and needed
        if not self.is_enabled():
            return text
        
        confidence_threshold = self.config.get('correction', {}).get('confidence_threshold', 0.6)
        if confidence >= confidence_threshold:
            logger.debug(f"Skipping LLM correction (confidence {confidence:.3f} >= {confidence_threshold})")
            return text
        
        if not text.strip():
            return text
        
        # Limit context window
        context_window = self.config.get('correction', {}).get('context_window', 200)
        if len(context) > context_window:
            context = context[:context_window] + "..."
        
        try:
            corrected = self.provider.correct_text(text, context)
            
            # Basic validation - corrected text shouldn't be drastically different
            if len(corrected) > len(text) * 3 or len(corrected) < len(text) * 0.3:
                logger.warning("LLM correction result seems invalid, using original text")
                return text
            
            logger.debug(f"Text corrected: '{text[:50]}...' -> '{corrected[:50]}...'")
            return corrected
            
        except Exception as e:
            error_logger.log_llm_error(e, self.config.get('provider', 'unknown'), len(text), "text correction")
            return text
    
    def is_enabled(self) -> bool:
        """Check if LLM correction is enabled and available."""
        return (self.config.get('correction', {}).get('enabled', False) and 
                self.provider is not None and 
                self.provider.is_available())
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        if not self.provider:
            return {'provider': 'none', 'available': False}
        
        return {
            'provider': self.config.get('provider', 'unknown'),
            'available': self.provider.is_available(),
            'config': self.config
        }
