try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except Exception:
    detect = None

class LanguageDetector:
    SUPPORTED_LANGUAGES = ['de', 'en', 'fr', 'it', 'es']
    DEFAULT_LANGUAGE = 'en'

    def detect_language(self, text):
        if detect:
            try:
                lang = detect(text)
                if lang in self.SUPPORTED_LANGUAGES:
                    return lang
            except Exception:
                pass
        return self.DEFAULT_LANGUAGE
