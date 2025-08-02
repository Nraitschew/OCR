# OCR System - Dateiübersicht

## 🚀 Setup-Skripte

### CPU-Only (Empfohlen)
- **`setup_cpu_only.sh`** - Installiert nur CPU-basierte Abhängigkeiten
- **`run_cpu_tests.sh`** - Führt CPU-Tests aus (wird automatisch erstellt)
- **`start_cpu_ocr.sh`** - Startet interaktive CPU-OCR-Session

### Vollständiges System (mit optionaler GPU)
- **`auto_setup.sh`** - Automatisches Setup für das vollständige System
- **`setup_and_test.sh`** - Setup + Tests in einem Schritt
- **`setup.sh`** - Basis-Setup-Skript

## 🐍 OCR-Systeme

### CPU-Only Version
- **`cpu_only_ocr_system.py`** - CPU-optimiertes OCR-System
  - Verwendet nur Tesseract
  - Ressourcen-Management
  - Bildvorverarbeitung
  - Batch-Verarbeitung

### Erweiterte Version
- **`enhanced_ocr_system.py`** - Erweitertes System mit Tabellenerkennung
- **`minimal_ocr_system.py`** - Minimale funktionsfähige Version
- **`ocr_system.py`** - Original-System mit GPU-Support

## 🧪 Test-Systeme

### Test-Runner
- **`test_cpu_only_system.py`** - CPU-spezifische Tests
- **`run_all_tests.py`** - Umfassende Tests für erweitertes System
- **`run_comprehensive_test.py`** - Detaillierte Tests mit Farbausgabe
- **`test_ocr_system.py`** - Basis-Testsuite

### Test-Dokument-Generatoren
- **`test_document_generator.py`** - Erstellt Basis-Testdokumente
- **`enhanced_test_generator.py`** - Erstellt komplexe Testdokumente
  - Rechnungen
  - Wissenschaftliche Dokumente
  - Finanzberichte
  - Verschachtelte Tabellen

## 📋 Anforderungen

- **`requirements.txt`** - Vollständige Abhängigkeiten (inkl. GPU)
- **`requirements_cpu.txt`** - CPU-only Abhängigkeiten (empfohlen)

## 📁 Verzeichnisse

- **`test_documents/`** - Basis-Testdokumente
- **`test_documents_enhanced/`** - Erweiterte Testdokumente
- **`venv/`** - Python Virtual Environment (vollständig)
- **`venv_cpu/`** - CPU-only Virtual Environment

## 📄 Dokumentation

- **`README.md`** - Hauptdokumentation
- **`CPU_ONLY_SUMMARY.md`** - CPU-System Zusammenfassung
- **`SYSTEM_OVERVIEW.md`** - Systemübersicht
- **`TEST_REPORT.md`** - Testbericht
- **`FILES_OVERVIEW.md`** - Diese Datei

## 🔧 Hilfsskripte

- **`simple_test.py`** - Einfacher Funktionstest
- **`test_results.json`** - Detaillierte Testergebnisse (automatisch generiert)

## 💡 Empfohlene Nutzung

### Für CPU-Only (empfohlen):
```bash
# 1. Setup
./setup_cpu_only.sh

# 2. Tests
./run_cpu_tests.sh

# 3. Nutzung
source venv_cpu/bin/activate
python3 your_script.py
```

### Für vollständiges System:
```bash
# 1. Setup + Test
./setup_and_test.sh

# 2. Nutzung
source venv/bin/activate
python3 your_script.py
```

## 🎯 Welche Datei für was?

- **Schnellstart CPU**: `setup_cpu_only.sh`
- **Vollständige Installation**: `auto_setup.sh`
- **Eigene Skripte**: Importiere `cpu_only_ocr_system.py`
- **Tests durchführen**: `test_cpu_only_system.py`
- **Testdokumente erstellen**: `enhanced_test_generator.py`