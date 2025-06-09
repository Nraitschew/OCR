import magic

class DocumentAnalyzer:
    def analyze_document_type(self, file_path):
        mime = magic.from_file(file_path, mime=True)
        return mime
