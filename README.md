\
# StudyMate AI

A laptop-only AI/ML Engineers' Day project.

## Features

- PDF upload
- Semantic search using SentenceTransformers + FAISS
- RAG-based Q&A grounded in the uploaded PDF
- Multiple explanation levels
- AI summary
- Important-topic analysis
- AI quiz with score
- Flashcards
- Personalized study plan
- Local Ollama model support

## 1. Install Python

Use Python 3.10, 3.11, or 3.12.

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows CMD:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Install Ollama

Install Ollama from its official website.

Then open a terminal and download the local model:

```bash
ollama pull gemma3:4b
```

If your laptop has limited RAM, use:

```bash
ollama pull gemma3:1b
```

Then set `gemma3:1b` in the app sidebar.

The first SentenceTransformer model download also requires internet.
After the models are downloaded, the main project can run locally.

## 4. Start the app

Make sure Ollama is running, then:

```bash
streamlit run app.py
```

Or on Windows, double-click:

```text
run_windows.bat
```

## Suggested live demo

1. Upload a text-based PDF.
2. Ask a question from the PDF.
3. Show the retrieved evidence.
4. Generate important topics.
5. Generate and submit a quiz.
6. Show the personalized study plan.
7. Open the "How it works" tab and explain the RAG pipeline.

## Competition title

**StudyMate AI — Intelligent Personal Learning Assistant**

Tagline: **Upload. Understand. Practice. Improve.**

## Notes

- Scanned/image-only PDFs may not contain selectable text and are not OCR'd in this version.
- The project does not claim to predict exact exam questions.
- For a stronger demo, use a 10–30 page PDF with clear text.

## Android app (Flutter)
The Flutter mobile app is in this repository alongside the Streamlit app. To build an Android APK, install Flutter and run `flutter build apk --release` from the Flutter project directory.
