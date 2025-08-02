"""
Language detection module with German umlaut support
"""
import logging
from typing import List, Dict, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Language detection with special support for German umlauts"""
    
    def __init__(self):
        self.detector = None
        self._initialize_detector()
        
        # German umlaut patterns
        self.german_chars = {
            'umlauts': ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü'],
            'eszett': ['ß'],
            'common_words': [
                'der', 'die', 'das', 'und', 'ist', 'ein', 'eine', 'für',
                'mit', 'auf', 'von', 'zu', 'den', 'des', 'dem', 'durch'
            ]
        }
        
    def _initialize_detector(self):
        """Initialize the language detection library"""
        try:
            # Try fast-langdetect first (fastest option)
            from fast_langdetect import detect, detect_langs
            self.detect = detect
            self.detect_langs = detect_langs
            self.detector_type = 'fast-langdetect'
            logger.info("Using fast-langdetect for language detection")
            
        except ImportError:
            try:
                # Fallback to langdetect
                from langdetect import detect, detect_langs
                self.detect = detect
                self.detect_langs = detect_langs
                self.detector_type = 'langdetect'
                logger.info("Using langdetect for language detection")
                
            except ImportError:
                logger.error("No language detection library available. Install fast-langdetect or langdetect")
                self.detector_type = None
                
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of text
        
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text or not text.strip():
            return ('unknown', 0.0)
            
        # Check for German indicators first
        german_score = self._check_german_indicators(text)
        
        if self.detector_type:
            try:
                # Get language predictions
                if self.detector_type == 'fast-langdetect':
                    # fast-langdetect returns (lang, score) tuple
                    result = self.detect(text)
                    if isinstance(result, tuple):
                        lang, confidence = result
                    else:
                        lang = result
                        confidence = 1.0
                else:
                    # langdetect
                    lang = self.detect(text)
                    confidence = 0.95  # langdetect doesn't provide confidence
                    
                # Boost German confidence if umlauts detected
                if lang == 'de' and german_score > 0:
                    confidence = min(1.0, confidence + german_score * 0.1)
                elif german_score > 0.5 and lang != 'de':
                    # Strong German indicators but different language detected
                    logger.warning(f"German indicators found but detected as {lang}")
                    
                return (lang, confidence)
                
            except Exception as e:
                logger.error(f"Language detection failed: {e}")
                
        # Fallback to German detection if no detector available
        if german_score > 0.3:
            return ('de', german_score)
            
        return ('unknown', 0.0)
        
    def detect_multiple_languages(self, text: str) -> List[Dict[str, float]]:
        """
        Detect multiple languages in text
        
        Returns:
            List of dictionaries with language and probability
        """
        if not text or not text.strip():
            return []
            
        results = []
        
        if self.detector_type:
            try:
                # Get multiple language predictions
                langs = self.detect_langs(text)
                
                if self.detector_type == 'fast-langdetect':
                    # Process fast-langdetect results
                    for lang_info in langs:
                        if isinstance(lang_info, tuple):
                            lang, prob = lang_info
                            results.append({'language': lang, 'probability': prob})
                        else:
                            results.append({'language': str(lang_info), 'probability': 0.5})
                else:
                    # Process langdetect results
                    for lang_info in langs:
                        results.append({
                            'language': lang_info.lang,
                            'probability': lang_info.prob
                        })
                        
            except Exception as e:
                logger.error(f"Multiple language detection failed: {e}")
                
        # Add German if umlauts detected but not in results
        german_score = self._check_german_indicators(text)
        if german_score > 0.3:
            german_found = any(r['language'] == 'de' for r in results)
            if not german_found:
                results.append({'language': 'de', 'probability': german_score})
                
        return sorted(results, key=lambda x: x['probability'], reverse=True)
        
    def _check_german_indicators(self, text: str) -> float:
        """
        Check for German language indicators
        
        Returns:
            Score between 0 and 1 indicating likelihood of German
        """
        if not text:
            return 0.0
            
        text_lower = text.lower()
        score = 0.0
        max_score = 0.0
        
        # Check for umlauts
        umlaut_count = sum(text.count(char) for char in self.german_chars['umlauts'])
        if umlaut_count > 0:
            score += min(0.4, umlaut_count * 0.05)
            max_score += 0.4
            
        # Check for eszett
        eszett_count = sum(text.count(char) for char in self.german_chars['eszett'])
        if eszett_count > 0:
            score += min(0.2, eszett_count * 0.1)
            max_score += 0.2
            
        # Check for common German words
        word_matches = sum(1 for word in self.german_chars['common_words'] 
                          if f' {word} ' in f' {text_lower} ')
        if word_matches > 0:
            score += min(0.4, word_matches * 0.05)
            max_score += 0.4
            
        # Normalize score
        return score / max_score if max_score > 0 else 0.0
        
    def validate_german_text(self, text: str) -> Dict[str, any]:
        """
        Validate German text and check for proper umlaut handling
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'has_umlauts': False,
            'has_eszett': False,
            'umlaut_count': 0,
            'eszett_count': 0,
            'umlaut_positions': [],
            'properly_encoded': True,
            'encoding_issues': []
        }
        
        # Check for umlauts
        for i, char in enumerate(text):
            if char in self.german_chars['umlauts']:
                results['has_umlauts'] = True
                results['umlaut_count'] += 1
                results['umlaut_positions'].append({
                    'position': i,
                    'character': char,
                    'context': text[max(0, i-5):i+6]
                })
                
        # Check for eszett
        for char in self.german_chars['eszett']:
            count = text.count(char)
            if count > 0:
                results['has_eszett'] = True
                results['eszett_count'] += count
                
        # Check for common encoding issues
        encoding_patterns = [
            (r'Ã¤', 'ä'), (r'Ã¶', 'ö'), (r'Ã¼', 'ü'),
            (r'Ã\x84', 'Ä'), (r'Ã\x96', 'Ö'), (r'Ã\x9c', 'Ü'),
            (r'ÃŸ', 'ß'), (r'Ã\x9f', 'ß')
        ]
        
        for pattern, correct in encoding_patterns:
            if pattern in text:
                results['properly_encoded'] = False
                results['encoding_issues'].append({
                    'found': pattern,
                    'should_be': correct,
                    'count': text.count(pattern)
                })
                
        return results
        
    def get_language_name(self, code: str) -> str:
        """Get full language name from ISO code"""
        language_names = {
            'de': 'German',
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'pl': 'Polish',
            'ru': 'Russian',
            'ja': 'Japanese',
            'zh': 'Chinese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'tr': 'Turkish'
        }
        return language_names.get(code, code.upper())