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

Then change `DEFAULT_MODEL` to `"gemma3:1b"` in `modules/ollama_client.py`
(the app's sidebar is hidden, so the model is set in code).

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

## Optional: cloud AI with Groq (no Ollama needed)

If a `GROQ_API_KEY` is available, the app uses Groq's `openai/gpt-oss-20b`
model instead of local Ollama. This is how a hosted deployment (for example
Streamlit Community Cloud) can run without a local model.

- Locally: set the `GROQ_API_KEY` environment variable, or add it to
  `.streamlit/secrets.toml`:

  ```toml
  GROQ_API_KEY = "your-key-here"
  ```

- On Streamlit Community Cloud: add `GROQ_API_KEY` under the app's Secrets.

Never commit your API key to the repository. A template is provided in
`.streamlit/secrets.toml.example`.

## Deploy and share a link (Streamlit Community Cloud)

1. Get a free API key at <https://console.groq.com>.
2. Go to <https://share.streamlit.io>, sign in with GitHub and choose
   **Create app**.
3. Repository: `fabrikaroxman/StudyMate-AI`, branch `main`,
   main file path `app.py`.
4. Open **Advanced settings**, pick Python 3.12, and paste this into
   **Secrets**:

   ```toml
   GROQ_API_KEY = "your-key-here"
   ```

5. Click **Deploy**. You get a public `https://<name>.streamlit.app` link
   to share.

Anyone with the link uses your Groq quota. If the app runs out of memory on
the free tier, check the app logs: `sentence-transformers` and `faiss-cpu`
are the heaviest dependencies.

## Running the tests

The tests cover text chunking and quiz validation and need no model, no
internet and no PDF library:

```bash
pip install pytest
python -m pytest
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

The Flutter app (`lib/main.dart`, `android/`) is a thin wrapper: it opens the
deployed Streamlit app in a WebView. It does not run any AI itself, so it only
works while that deployment is online. To point it at your own deployment,
change the URL in `lib/main.dart`.

To build an APK, install Flutter (a version supporting Dart SDK `^3.13.3`)
and run this from the repository root:

```bash
flutter pub get
flutter build apk --release
```
