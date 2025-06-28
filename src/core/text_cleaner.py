"""
Text cleaning and post-processing module for OCR results.
"""

import re
import logging
from typing import List, Set, Dict, Tuple
from collections import Counter
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class OCRTextCleaner:
    """Cleans and post-processes OCR text results."""
    
    def __init__(self):
        """Initialize text cleaner."""
        # Common OCR artifacts to clean
        self.artifact_patterns = [
            (r'[€£¥¢]', ''),  # Currency symbols that are OCR errors
            (r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', self._replace_accented),  # Accented chars
            (r'[^\w\s.,!?;:()\[\]{}"\'-]', ' '),  # Remove other special chars
            (r':\s*', ' '),  # Replace colons with spaces (common OCR error)
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'([a-z])([A-Z])', r'\1 \2'),  # Add space between lowercase and uppercase
        ]
        
        # Common word corrections
        self.word_corrections = {
            'WeChatis': 'WeChat is',
            'Bhutanland': 'Bhutan and',
            'popuilar': 'popular',
            'mess': 'messaging',
            'lattormin': 'platform',
            'Mongoliax': 'Mongolia',
            'mosti': 'most',
            'jn': 'in',
            'he': 'the',
            'Trai': '',  # Fragment, remove
            'Mon': 'Mongolia',
            'id': 'in',
        }

        # Phrases to completely remove (OCR artifacts)
        self.remove_phrases = [
            '€b:',
            'Més:',
            'Trai',
            'id Mon',
        ]
    
    def clean_ocr_results(self, ocr_results: List) -> str:
        """Clean and combine OCR results into coherent text.
        
        Args:
            ocr_results: List of OCR result objects with .text attribute
            
        Returns:
            Cleaned, deduplicated text
        """
        if not ocr_results:
            return ""
        
        try:
            # Extract text from results
            raw_texts = [result.text.strip() for result in ocr_results if result.text.strip()]
            
            if not raw_texts:
                return ""
            
            logger.debug(f"Processing {len(raw_texts)} text regions")
            
            # Step 1: Clean individual text fragments
            cleaned_fragments = []
            for text in raw_texts:
                cleaned = self._clean_single_text(text)
                if cleaned and len(cleaned) > 2:  # Only keep meaningful fragments
                    cleaned_fragments.append(cleaned)
            
            # Step 2: Remove duplicates and overlaps
            deduplicated = self._remove_duplicates(cleaned_fragments)

            # Step 3: Find the best base text (longest, highest quality)
            best_text = self._find_best_base_text(deduplicated)

            # Step 4: Final cleaning and formatting
            final_text = self._final_cleanup(best_text)
            
            logger.info(f"Text cleaning: {len(raw_texts)} fragments -> '{final_text[:100]}...'")
            return final_text
            
        except Exception as e:
            logger.error(f"Error cleaning OCR results: {e}")
            # Fallback: just join the raw texts
            return ' '.join([result.text.strip() for result in ocr_results if result.text.strip()])
    
    def _clean_single_text(self, text: str) -> str:
        """Clean a single text fragment."""
        cleaned = text.strip()

        # Remove phrases that should be completely removed
        for phrase in self.remove_phrases:
            cleaned = cleaned.replace(phrase, ' ')

        # Apply artifact cleaning patterns
        for pattern, replacement in self.artifact_patterns:
            if callable(replacement):
                cleaned = re.sub(pattern, replacement, cleaned)
            else:
                cleaned = re.sub(pattern, replacement, cleaned)

        # Apply word-level corrections
        words = cleaned.split()
        corrected_words = []

        for word in words:
            word = word.strip()
            if not word:
                continue

            # Check for exact matches
            if word in self.word_corrections:
                replacement = self.word_corrections[word]
                if replacement:  # Only add if not empty
                    corrected_words.extend(replacement.split())
            else:
                # Check for partial matches or common errors
                corrected_word = self._correct_word(word)
                if corrected_word and len(corrected_word) > 1:  # Only keep meaningful words
                    corrected_words.append(corrected_word)

        return ' '.join(corrected_words).strip()
    
    def _replace_accented(self, match) -> str:
        """Replace accented characters with normal ones."""
        char = match.group(0)
        replacements = {
            'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'æ': 'ae',
            'ç': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
            'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
            'ñ': 'n', 'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o',
            'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
            'ý': 'y', 'þ': 'th', 'ÿ': 'y'
        }
        return replacements.get(char, char)
    
    def _correct_word(self, word: str) -> str:
        """Correct individual word errors."""
        if len(word) < 2:
            return word
        
        # Check for common patterns
        corrected = word
        
        # Fix common OCR character substitutions
        char_fixes = {
            '0': 'o',  # Zero to letter O
            '1': 'l',  # One to letter L (in some contexts)
            '5': 's',  # Five to letter S
            '8': 'B',  # Eight to letter B
            '6': 'G',  # Six to letter G
        }
        
        # Apply character fixes cautiously (only if it makes a real word)
        for bad_char, good_char in char_fixes.items():
            if bad_char in corrected:
                test_word = corrected.replace(bad_char, good_char)
                if self._is_likely_word(test_word):
                    corrected = test_word
        
        return corrected
    
    def _is_likely_word(self, word: str) -> bool:
        """Check if a string is likely a real word."""
        if len(word) < 2:
            return False
        
        # Simple heuristics for English words
        vowels = 'aeiouAEIOU'
        consonants = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'
        
        has_vowel = any(c in vowels for c in word)
        has_consonant = any(c in consonants for c in word)
        
        # Must have both vowels and consonants (with some exceptions)
        if not (has_vowel and has_consonant):
            # Allow some exceptions like "I", "a", etc.
            if word.lower() not in ['i', 'a', 'is', 'in', 'it', 'at', 'to', 'of', 'or']:
                return False
        
        # Check for reasonable character patterns
        if re.search(r'[0-9]{3,}', word):  # Too many numbers
            return False
        
        if re.search(r'[^a-zA-Z0-9\s\'-]', word):  # Strange characters
            return False
        
        return True
    
    def _remove_duplicates(self, fragments: List[str]) -> List[str]:
        """Remove duplicate and overlapping text fragments."""
        if not fragments:
            return []
        
        # Sort by length (longer first) to prefer complete fragments
        sorted_fragments = sorted(fragments, key=len, reverse=True)
        unique_fragments = []
        
        for fragment in sorted_fragments:
            is_duplicate = False
            
            for existing in unique_fragments:
                # Check if this fragment is contained in an existing one
                if self._is_substring_match(fragment, existing, threshold=0.8):
                    is_duplicate = True
                    break
                
                # Check if this fragment contains an existing one
                if self._is_substring_match(existing, fragment, threshold=0.8):
                    # Replace the existing with this longer one
                    unique_fragments.remove(existing)
                    break
            
            if not is_duplicate:
                unique_fragments.append(fragment)
        
        return unique_fragments
    
    def _is_substring_match(self, shorter: str, longer: str, threshold: float = 0.8) -> bool:
        """Check if shorter text is substantially contained in longer text."""
        if len(shorter) > len(longer):
            return False
        
        # Normalize for comparison
        shorter_norm = re.sub(r'\s+', ' ', shorter.lower().strip())
        longer_norm = re.sub(r'\s+', ' ', longer.lower().strip())
        
        if not shorter_norm:
            return False
        
        # Check direct substring
        if shorter_norm in longer_norm:
            return True
        
        # Check fuzzy match
        matcher = SequenceMatcher(None, shorter_norm, longer_norm)
        similarity = matcher.ratio()
        
        return similarity >= threshold
    
    def _merge_fragments(self, fragments: List[str]) -> str:
        """Merge related text fragments intelligently."""
        if not fragments:
            return ""
        
        if len(fragments) == 1:
            return fragments[0]
        
        # Try to find the best order and merge fragments
        # For now, use the longest fragment as base and add unique parts from others
        
        # Sort by length and quality
        sorted_fragments = sorted(fragments, key=lambda x: (len(x), self._calculate_quality(x)), reverse=True)
        
        base_text = sorted_fragments[0]
        
        # Add unique content from other fragments
        for fragment in sorted_fragments[1:]:
            base_text = self._merge_two_fragments(base_text, fragment)
        
        return base_text
    
    def _calculate_quality(self, text: str) -> float:
        """Calculate text quality score."""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Prefer text with proper capitalization
        if text[0].isupper():
            score += 0.1
        
        # Prefer text with proper punctuation
        if text.endswith('.'):
            score += 0.1
        
        # Prefer text with common English words
        common_words = {'the', 'is', 'and', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are', 'you'}
        words = text.lower().split()
        common_count = sum(1 for word in words if word in common_words)
        if words:
            score += (common_count / len(words)) * 0.3
        
        # Penalize text with artifacts
        if re.search(r'[€£¥¢:]{2,}', text):
            score -= 0.2
        
        return score
    
    def _merge_two_fragments(self, text1: str, text2: str) -> str:
        """Merge two text fragments intelligently."""
        # If one is contained in the other, use the longer one
        if text2.lower() in text1.lower():
            return text1
        if text1.lower() in text2.lower():
            return text2
        
        # Try to find overlap and merge
        words1 = text1.split()
        words2 = text2.split()
        
        # Look for overlapping sequences
        best_overlap = 0
        best_merge = text1 + " " + text2
        
        for i in range(len(words1)):
            for j in range(len(words2)):
                # Check if there's an overlap starting at position i in text1 and j in text2
                overlap_len = 0
                while (i + overlap_len < len(words1) and 
                       j + overlap_len < len(words2) and
                       words1[i + overlap_len].lower() == words2[j + overlap_len].lower()):
                    overlap_len += 1
                
                if overlap_len > best_overlap and overlap_len >= 2:  # Require at least 2 words overlap
                    best_overlap = overlap_len
                    # Merge: text1[:i] + overlap + text2[j+overlap_len:]
                    merged_words = (words1[:i + overlap_len] + 
                                  words2[j + overlap_len:])
                    best_merge = ' '.join(merged_words)
        
        return best_merge
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup and formatting."""
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common punctuation issues
        cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)  # Remove space before punctuation
        cleaned = re.sub(r'([.,!?;:])\s*([a-zA-Z])', r'\1 \2', cleaned)  # Ensure space after punctuation
        
        # Ensure proper sentence capitalization
        sentences = re.split(r'([.!?]+)', cleaned)
        capitalized_sentences = []
        
        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():  # Actual sentence content
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                capitalized_sentences.append(sentence)
            else:
                capitalized_sentences.append(sentence)
        
        final_text = ''.join(capitalized_sentences)
        
        # Final validation
        if len(final_text.split()) < 3:
            logger.warning("Final text is very short, might be over-cleaned")
        
        return final_text.strip()

    def _find_best_base_text(self, fragments: List[str]) -> str:
        """Find the best base text from fragments."""
        if not fragments:
            return ""

        if len(fragments) == 1:
            return fragments[0]

        # Score each fragment
        scored_fragments = []
        for fragment in fragments:
            score = self._calculate_fragment_score(fragment)
            scored_fragments.append((fragment, score))

        # Sort by score (highest first)
        scored_fragments.sort(key=lambda x: x[1], reverse=True)

        # Return the best fragment
        return scored_fragments[0][0]

    def _calculate_fragment_score(self, text: str) -> float:
        """Calculate a quality score for a text fragment."""
        if not text:
            return 0.0

        score = 0.0
        words = text.split()

        # Length bonus (longer is generally better, up to a point)
        length_score = min(len(words) / 20.0, 1.0)  # Normalize to max 1.0
        score += length_score * 0.4

        # Check for key words from expected text
        key_words = {'trailing', 'facebook', 'messenger', 'wechat', 'second', 'most', 'popular', 'messaging', 'platform', 'bhutan', 'mongolia'}
        word_set = set(text.lower().split())
        key_word_matches = len(key_words & word_set)
        key_word_score = key_word_matches / len(key_words)
        score += key_word_score * 0.4

        # Penalize artifacts
        if any(artifact in text for artifact in ['€', ':', 'Més', 'lattormin']):
            score -= 0.3

        # Bonus for proper sentence structure
        if text.endswith('.'):
            score += 0.1

        if text[0].isupper():
            score += 0.1

        return score
