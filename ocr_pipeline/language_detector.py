from langdetect import detect

class LanguageDetector:
    SUPPORTED_LANGUAGES = ['de', 'en', 'fr', 'it', 'es']
    DEFAULT_LANGUAGE = 'en'

    def detect_language(self, text):
        try:
            lang = detect(text)
        except Exception:
            return self.DEFAULT_LANGUAGE
        if lang not in self.SUPPORTED_LANGUAGES:
            return self.DEFAULT_LANGUAGE
        return lang
