try:
    import magic
except Exception:
    magic = None
import mimetypes
from pathlib import Path

class DocumentAnalyzer:
    def analyze_document_type(self, file_path):
        if magic:
            try:
                return magic.from_file(str(file_path), mime=True)
            except Exception:
                pass
        mime, _ = mimetypes.guess_type(str(file_path))
        return mime or 'application/octet-stream'
