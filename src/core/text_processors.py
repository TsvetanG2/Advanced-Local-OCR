"""Text processing and comparison module."""

import re
import unicodedata
import difflib
from typing import List, Dict, Any, Tuple, Optional
import logging
from Levenshtein import distance as lev_distance

from ..utils.config import get_config

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Handles text normalization for comparison."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize text normalizer.
        
        Args:
            config: Text processing configuration. If None, uses global config.
        """
        self.config = config or get_config().get('text_processing', {})
        self.normalization_config = self.config.get('normalization', {})
    
    def normalize(self, text: str, preserve_case: bool = False) -> str:
        """Normalize text for comparison.
        
        Args:
            text: Input text
            preserve_case: Whether to preserve original case
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        try:
            # Unicode normalization
            unicode_form = self.normalization_config.get('unicode_form', 'NFKC')
            normalized = unicodedata.normalize(unicode_form, text)
            
            # Replace common quote variations
            normalized = self._normalize_quotes(normalized)
            
            # Preserve specific patterns before general cleanup
            preserved_patterns = self._preserve_patterns(normalized)
            
            # Remove punctuation if configured
            if self.normalization_config.get('remove_punctuation', True):
                normalized = re.sub(r'[^\w\s]', ' ', preserved_patterns['text'])
            else:
                normalized = preserved_patterns['text']
            
            # Normalize whitespace
            normalized = ' '.join(normalized.split())
            
            # Restore preserved patterns
            normalized = self._restore_patterns(normalized, preserved_patterns['patterns'])
            
            # Case normalization
            if not preserve_case:
                normalized = normalized.lower()
            
            return normalized.strip()
            
        except Exception as e:
            logger.error(f"Error normalizing text: {e}")
            return text
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize different quote characters."""
        # Replace various quote characters with standard ones
        quote_replacements = {
            '„': '"',  # German-style opening quote
            '"': '"',  # Closing quote
            ''': "'",  # Smart single quote
            ''': "'",  # Smart single quote
            '`': "'",  # Backtick
        }
        
        for old, new in quote_replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _preserve_patterns(self, text: str) -> Dict[str, Any]:
        """Preserve specific patterns during normalization.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with processed text and preserved patterns
        """
        patterns_to_preserve = self.normalization_config.get('preserve_patterns', [])
        preserved = {}
        processed_text = text
        
        for pattern_type in patterns_to_preserve:
            if pattern_type == 'email_addresses':
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                emails = re.findall(email_pattern, processed_text)
                for i, email in enumerate(emails):
                    placeholder = f"EMAIL_PLACEHOLDER_{i}"
                    preserved[placeholder] = email
                    processed_text = processed_text.replace(email, placeholder)
            
            elif pattern_type == 'urls':
                url_pattern = r'https?://[^\s]+'
                urls = re.findall(url_pattern, processed_text)
                for i, url in enumerate(urls):
                    placeholder = f"URL_PLACEHOLDER_{i}"
                    preserved[placeholder] = url
                    processed_text = processed_text.replace(url, placeholder)
            
            elif pattern_type == 'legal_refs':
                # Future: Add legal reference patterns for other languages
                # Currently focused on English, but extensible
                legal_pattern = r'\b(Art|Article|Sec|Section|Para|Paragraph)\.?\s*\d+[a-z]?\b'
                legal_refs = re.findall(legal_pattern, processed_text, re.IGNORECASE)
                for i, ref in enumerate(legal_refs):
                    placeholder = f"LEGAL_PLACEHOLDER_{i}"
                    preserved[placeholder] = ref
                    processed_text = processed_text.replace(ref, placeholder)
        
        return {
            'text': processed_text,
            'patterns': preserved
        }
    
    def _restore_patterns(self, text: str, patterns: Dict[str, str]) -> str:
        """Restore preserved patterns after normalization."""
        for placeholder, original in patterns.items():
            text = text.replace(placeholder, original)
        return text


class TextTokenizer:
    """Handles intelligent text tokenization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize text tokenizer.
        
        Args:
            config: Text processing configuration. If None, uses global config.
        """
        self.config = config or get_config().get('text_processing', {})
        self.language = get_config().get_current_language()
    
    def tokenize(self, text: str, mode: str = 'standard') -> List[str]:
        """Tokenize text for comparison.
        
        Args:
            text: Input text
            mode: Tokenization mode ('standard', 'legal', 'technical')
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        try:
            if mode == 'legal':
                return self._tokenize_legal(text)
            elif mode == 'technical':
                return self._tokenize_technical(text)
            else:
                return self._tokenize_standard(text)
                
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return text.split()
    
    def _tokenize_standard(self, text: str) -> List[str]:
        """Standard tokenization by whitespace."""
        return text.split()
    
    def _tokenize_legal(self, text: str) -> List[str]:
        """Legal document tokenization (extensible for different languages)."""
        tokens = []
        
        # English legal patterns
        if self.language == 'en':
            legal_pattern = r'(\s|^)(Art|Article|Sec|Section|Para|Paragraph)\.?\s*(\d+[a-z]?)(\s|$|,|\.)'
            
            last_end = 0
            for match in re.finditer(legal_pattern, text, re.IGNORECASE):
                # Add text before match
                if match.start() > last_end:
                    before_text = text[last_end:match.start()].strip()
                    if before_text:
                        tokens.extend(before_text.split())
                
                # Add legal reference as single token
                legal_ref = match.group(2) + match.group(3)
                tokens.append(legal_ref)
                last_end = match.end() - len(match.group(4))
            
            # Add remaining text
            if last_end < len(text):
                remaining = text[last_end:].strip()
                if remaining:
                    tokens.extend(remaining.split())
        
        # Future: Add other language patterns here
        # elif self.language == 'bg':
        #     # Bulgarian legal patterns
        #     pass
        
        return tokens if tokens else text.split()
    
    def _tokenize_technical(self, text: str) -> List[str]:
        """Technical document tokenization."""
        tokens = []
        
        # Preserve technical terms, version numbers, etc.
        technical_pattern = r'(\s|^)(v\d+\.\d+|\d+\.\d+\.\d+|[A-Z]{2,}[0-9]+)(\s|$|,|\.)'
        
        last_end = 0
        for match in re.finditer(technical_pattern, text):
            # Add text before match
            if match.start() > last_end:
                before_text = text[last_end:match.start()].strip()
                if before_text:
                    tokens.extend(before_text.split())
            
            # Add technical term as single token
            tech_term = match.group(2)
            tokens.append(tech_term)
            last_end = match.end() - len(match.group(3))
        
        # Add remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                tokens.extend(remaining.split())
        
        return tokens if tokens else text.split()


class TextComparator:
    """Handles text comparison and error detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize text comparator.
        
        Args:
            config: Text processing configuration. If None, uses global config.
        """
        self.config = config or get_config().get('text_processing', {})
        self.comparison_config = self.config.get('comparison', {})
        self.normalizer = TextNormalizer(config)
        self.tokenizer = TextTokenizer(config)
    
    def compare_texts(self, expected: str, extracted: str, 
                     tokenization_mode: str = 'standard') -> Dict[str, Any]:
        """Compare expected and extracted texts.
        
        Args:
            expected: Expected text
            extracted: Extracted text
            tokenization_mode: Mode for tokenization
            
        Returns:
            Comparison results with errors and metrics
        """
        try:
            # Normalize texts
            expected_norm = self.normalizer.normalize(expected)
            extracted_norm = self.normalizer.normalize(extracted)
            
            # Tokenize texts
            expected_tokens = self.tokenizer.tokenize(expected_norm, tokenization_mode)
            extracted_tokens = self.tokenizer.tokenize(extracted_norm, tokenization_mode)
            
            # Perform comparison
            algorithm = self.comparison_config.get('algorithm', 'sequence_matcher')
            
            if algorithm == 'levenshtein':
                return self._compare_levenshtein(expected_tokens, extracted_tokens)
            else:
                return self._compare_sequence_matcher(expected_tokens, extracted_tokens)
                
        except Exception as e:
            logger.error(f"Error comparing texts: {e}")
            return {
                'errors': [f"Comparison failed: {str(e)}"],
                'similarity': 0.0,
                'metrics': {}
            }
    
    def _compare_sequence_matcher(self, expected_tokens: List[str], 
                                extracted_tokens: List[str]) -> Dict[str, Any]:
        """Compare using difflib.SequenceMatcher."""
        case_sensitive = self.comparison_config.get('case_sensitive', False)
        
        if not case_sensitive:
            expected_tokens = [t.lower() for t in expected_tokens]
            extracted_tokens = [t.lower() for t in extracted_tokens]
        
        matcher = difflib.SequenceMatcher(None, expected_tokens, extracted_tokens)
        errors = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            elif tag == 'delete':
                for token in expected_tokens[i1:i2]:
                    errors.append({
                        'type': 'missing',
                        'expected': token,
                        'found': None,
                        'message': f"Missing word: '{token}'"
                    })
            elif tag == 'insert':
                for token in extracted_tokens[j1:j2]:
                    errors.append({
                        'type': 'extra',
                        'expected': None,
                        'found': token,
                        'message': f"Extra word: '{token}'"
                    })
            elif tag == 'replace':
                for exp_token, ext_token in zip(expected_tokens[i1:i2], extracted_tokens[j1:j2]):
                    error_info = self._analyze_token_difference(exp_token, ext_token)
                    errors.append(error_info)
        
        # Calculate similarity
        similarity = matcher.ratio()
        
        # Calculate metrics
        metrics = self._calculate_metrics(expected_tokens, extracted_tokens, errors)
        
        return {
            'errors': errors,
            'similarity': similarity,
            'metrics': metrics
        }
    
    def _compare_levenshtein(self, expected_tokens: List[str], 
                           extracted_tokens: List[str]) -> Dict[str, Any]:
        """Compare using Levenshtein distance."""
        expected_text = ' '.join(expected_tokens)
        extracted_text = ' '.join(extracted_tokens)
        
        distance = lev_distance(expected_text, extracted_text)
        max_length = max(len(expected_text), len(extracted_text))
        similarity = 1 - (distance / max_length) if max_length > 0 else 1.0
        
        # For Levenshtein, we'll use a simplified error detection
        errors = []
        if similarity < self.comparison_config.get('similarity_threshold', 0.8):
            errors.append({
                'type': 'difference',
                'expected': expected_text,
                'found': extracted_text,
                'message': f"Text differs significantly (similarity: {similarity:.3f})"
            })
        
        metrics = {
            'levenshtein_distance': distance,
            'max_length': max_length,
            'expected_length': len(expected_text),
            'extracted_length': len(extracted_text)
        }
        
        return {
            'errors': errors,
            'similarity': similarity,
            'metrics': metrics
        }
    
    def _analyze_token_difference(self, expected: str, found: str) -> Dict[str, Any]:
        """Analyze the difference between two tokens."""
        # Calculate similarity
        token_similarity = 1 - (lev_distance(expected, found) / max(len(expected), len(found)))
        
        if token_similarity > 0.7:
            error_type = 'similar'
            message = f"Similar word: expected '{expected}', found '{found}'"
        else:
            error_type = 'different'
            message = f"Different word: expected '{expected}', found '{found}'"
        
        return {
            'type': error_type,
            'expected': expected,
            'found': found,
            'similarity': token_similarity,
            'message': message
        }
    
    def _calculate_metrics(self, expected_tokens: List[str], extracted_tokens: List[str], 
                         errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comparison metrics."""
        total_expected = len(expected_tokens)
        total_extracted = len(extracted_tokens)
        total_errors = len(errors)
        
        # Count error types
        error_counts = {}
        for error in errors:
            error_type = error['type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Calculate accuracy
        accuracy = 1 - (total_errors / max(total_expected, 1))
        
        return {
            'total_expected_tokens': total_expected,
            'total_extracted_tokens': total_extracted,
            'total_errors': total_errors,
            'error_counts': error_counts,
            'accuracy': accuracy,
            'precision': total_expected / max(total_extracted, 1) if total_extracted > 0 else 0,
            'recall': (total_expected - error_counts.get('missing', 0)) / max(total_expected, 1)
        }
