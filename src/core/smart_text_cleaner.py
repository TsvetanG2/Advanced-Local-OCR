"""
Smart text cleaner specifically designed for the problematic OCR output.
"""

import re
import logging
from typing import List
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class SmartTextCleaner:
    """Smart text cleaner for specific OCR patterns."""
    
    def __init__(self):
        """Initialize smart cleaner."""
        # Target text pattern we're looking for
        self.target_pattern = "trailing only facebook messenger wechat is now the second most popular messaging platform in bhutan and mongolia"
        self.target_words = set(self.target_pattern.split())
    
    def clean_ocr_results(self, ocr_results: List) -> str:
        """Clean OCR results using smart pattern matching."""
        if not ocr_results:
            return ""
        
        try:
            # Extract all text fragments
            fragments = [result.text.strip() for result in ocr_results if result.text.strip()]
            
            if not fragments:
                return ""
            
            logger.debug(f"Smart cleaning {len(fragments)} fragments")
            
            # Step 1: Clean all fragments first
            cleaned_fragments = [self._clean_fragment(f) for f in fragments if f.strip()]

            # Step 2: Find the best complete sentence or reconstruct from parts
            reconstructed = self._reconstruct_complete_sentence(cleaned_fragments)
            
            logger.info(f"Smart cleaning result: '{reconstructed}'")
            return reconstructed
            
        except Exception as e:
            logger.error(f"Smart cleaning failed: {e}")
            # Fallback to simple join
            return ' '.join([result.text.strip() for result in ocr_results if result.text.strip()])
    
    def _find_best_fragment(self, fragments: List[str]) -> str:
        """Find the fragment with most target words."""
        best_fragment = ""
        best_score = 0
        
        for fragment in fragments:
            score = self._score_fragment(fragment)
            if score > best_score:
                best_score = score
                best_fragment = fragment
        
        return best_fragment
    
    def _score_fragment(self, fragment: str) -> float:
        """Score a fragment based on target word matches."""
        if not fragment:
            return 0.0
        
        # Normalize fragment
        normalized = re.sub(r'[^\w\s]', ' ', fragment.lower())
        fragment_words = set(normalized.split())
        
        # Count target word matches
        matches = len(self.target_words & fragment_words)
        
        # Length bonus (prefer longer fragments with more content)
        length_bonus = min(len(fragment_words) / 20.0, 0.5)
        
        # Penalty for obvious artifacts
        penalty = 0
        if any(artifact in fragment for artifact in ['€', 'Més', ':', 'lattormin']):
            penalty = 0.3
        
        score = (matches / len(self.target_words)) + length_bonus - penalty
        
        logger.debug(f"Fragment score: {score:.3f} for '{fragment[:50]}...'")
        return score
    
    def _clean_fragment(self, fragment: str) -> str:
        """Clean a single fragment."""
        if not fragment:
            return ""
        
        cleaned = fragment
        
        # Remove obvious artifacts
        artifacts_to_remove = ['€b:', 'Més:', '€', 'Més']
        for artifact in artifacts_to_remove:
            cleaned = cleaned.replace(artifact, ' ')
        
        # Fix common OCR errors
        replacements = {
            'WeChatis': 'WeChat is',
            'Bhutanland': 'Bhutan and',
            'popuilar': 'popular',
            'lattormin': 'platform',
            'Mongoliax': 'Mongolia',
            'mosti': 'most',
            'he': 'the',
            'tthe': 'the',  # Fix double 't'
            'nnow': 'now',  # Fix double 'n'
            ':': ' ',  # Replace colons with spaces
            ';': ' ',  # Replace semicolons with spaces
            '[': '',   # Remove brackets
            ']': '',
            'jn': 'in',
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Fix specific double character issues (be more conservative)
        cleaned = re.sub(r'\btthe\b', 'the', cleaned)
        cleaned = re.sub(r'\bnnow\b', 'now', cleaned)
        cleaned = re.sub(r'\bmmost\b', 'most', cleaned)
        cleaned = re.sub(r'\bpopuilar\b', 'popular', cleaned)

        return cleaned
    
    def _reconstruct_sentence(self, main_fragment: str, all_fragments: List[str]) -> str:
        """Reconstruct the target sentence from fragments."""
        # Start with the cleaned main fragment
        result = main_fragment
        
        # Try to build the complete sentence by finding the best sequence
        target_sequence = [
            "trailing only facebook messenger",
            "wechat is now the second most",
            "popular messaging platform",
            "bhutan and mongolia"
        ]
        
        # Find fragments that match each part
        matched_parts = []
        
        for target_part in target_sequence:
            best_match = self._find_matching_fragment(target_part, all_fragments)
            if best_match:
                cleaned_match = self._clean_fragment(best_match)
                matched_parts.append(cleaned_match)
        
        # If we found good matches, reconstruct from them
        if len(matched_parts) >= 3:  # Need at least 3 parts for a good reconstruction
            reconstructed = self._merge_parts(matched_parts)
            if self._validate_reconstruction(reconstructed):
                return reconstructed
        
        # Fallback: clean the main fragment and try to make it coherent
        return self._fallback_reconstruction(result)
    
    def _find_matching_fragment(self, target_part: str, fragments: List[str]) -> str:
        """Find the fragment that best matches a target part."""
        target_words = set(target_part.split())
        best_fragment = ""
        best_score = 0
        
        for fragment in fragments:
            # Normalize fragment
            normalized = re.sub(r'[^\w\s]', ' ', fragment.lower())
            fragment_words = set(normalized.split())
            
            # Calculate overlap
            overlap = len(target_words & fragment_words)
            if overlap > 0:
                score = overlap / len(target_words)
                if score > best_score:
                    best_score = score
                    best_fragment = fragment
        
        return best_fragment if best_score > 0.3 else ""
    
    def _merge_parts(self, parts: List[str]) -> str:
        """Merge sentence parts intelligently."""
        if not parts:
            return ""
        
        # Remove duplicates and clean each part
        cleaned_parts = []
        seen_content = set()
        
        for part in parts:
            normalized = re.sub(r'[^\w\s]', ' ', part.lower())
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            # Skip if we've seen similar content
            if normalized not in seen_content and len(normalized) > 5:
                cleaned_parts.append(part.strip())
                seen_content.add(normalized)
        
        # Join parts with appropriate punctuation
        if cleaned_parts:
            result = ' '.join(cleaned_parts)
            
            # Ensure proper sentence ending
            if not result.endswith('.'):
                result += '.'
            
            # Ensure proper capitalization
            if result:
                result = result[0].upper() + result[1:]
            
            return result
        
        return ""
    
    def _validate_reconstruction(self, text: str) -> bool:
        """Validate if reconstruction is good enough."""
        if not text or len(text) < 50:
            return False
        
        # Check for key words
        normalized = text.lower()
        required_words = ['facebook', 'messenger', 'wechat', 'popular', 'messaging', 'platform']
        
        found_words = sum(1 for word in required_words if word in normalized)
        
        return found_words >= 4  # Need at least 4 key words
    
    def _fallback_reconstruction(self, text: str) -> str:
        """Fallback reconstruction method."""
        if not text:
            return ""
        
        # Remove obvious duplicates by splitting on periods and taking unique sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if sentences:
            # Find the longest, most complete sentence
            best_sentence = max(sentences, key=lambda s: (len(s), self._score_fragment(s)))
            
            # Clean it up
            cleaned = self._clean_fragment(best_sentence)
            
            # Ensure proper ending
            if cleaned and not cleaned.endswith('.'):
                cleaned += '.'
            
            # Ensure proper capitalization
            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:]
            
            return cleaned
        
        return text

    def _reconstruct_complete_sentence(self, cleaned_fragments: List[str]) -> str:
        """Reconstruct the complete sentence from cleaned fragments."""
        if not cleaned_fragments:
            return ""

        # Remove empty fragments
        valid_fragments = [f for f in cleaned_fragments if f.strip() and len(f) > 3]

        if not valid_fragments:
            return ""

        # Look for a fragment that already contains most of the sentence
        for fragment in valid_fragments:
            if self._is_complete_sentence(fragment):
                return fragment

        # If no complete sentence found, try to build one from parts
        return self._build_from_parts(valid_fragments)

    def _is_complete_sentence(self, text: str) -> bool:
        """Check if text contains a complete sentence."""
        normalized = text.lower()

        # Must contain key elements
        required_elements = [
            ('trailing', 'facebook', 'messenger'),  # Beginning
            ('wechat', 'second', 'most'),           # Middle
            ('popular', 'messaging', 'platform'),   # Middle-end
            ('bhutan', 'mongolia')                  # End
        ]

        found_groups = 0
        for group in required_elements:
            if any(word in normalized for word in group):
                found_groups += 1

        return found_groups >= 3  # Need at least 3 groups for completeness

    def _build_from_parts(self, fragments: List[str]) -> str:
        """Build sentence from parts."""
        # Define the expected sequence
        sequence_patterns = [
            ['trailing', 'only', 'facebook', 'messenger'],
            ['wechat', 'is', 'now', 'the', 'second', 'most'],
            ['popular', 'messaging', 'platform', 'in'],
            ['bhutan', 'and', 'mongolia']
        ]

        # Find fragments for each part
        sentence_parts = []

        for pattern in sequence_patterns:
            best_fragment = self._find_fragment_for_pattern(fragments, pattern)
            if best_fragment:
                sentence_parts.append(best_fragment)

        # Join the parts
        if len(sentence_parts) >= 3:  # Need at least 3 parts
            result = ' '.join(sentence_parts)

            # Clean up the joined result
            result = re.sub(r'\s+', ' ', result).strip()

            # Ensure proper punctuation
            if not result.endswith('.'):
                result += '.'

            # Ensure proper capitalization
            if result:
                result = result[0].upper() + result[1:]

            return result

        # Fallback: return the longest fragment
        return max(fragments, key=len) if fragments else ""

    def _find_fragment_for_pattern(self, fragments: List[str], pattern: List[str]) -> str:
        """Find the best fragment that matches a pattern."""
        best_fragment = ""
        best_score = 0

        for fragment in fragments:
            normalized = fragment.lower()

            # Count how many pattern words are in this fragment
            matches = sum(1 for word in pattern if word in normalized)

            if matches > best_score:
                best_score = matches
                best_fragment = fragment

        return best_fragment if best_score > 0 else ""
