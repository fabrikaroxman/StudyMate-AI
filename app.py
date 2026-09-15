\
import hashlib
import streamlit as st

from modules.pdf_reader import extract_pdf, split_text
from modules.vector_store import build_index, retrieve
from modules.ollama_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OllamaError,
    is_ollama_running,
)
from modules.features import (
    answer_question,
    summarize_material,
    extract_topics,
    make_flashcards,
    make_quiz,
    create_study_plan,
)


st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🎓",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container {max-width: 1200px; padding-top: 1.5rem;}
.hero {
    padding: 1.25rem 1.4rem;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 18px;
    margin-bottom: 1rem;
}
.small-muted {opacity: .72; font-size: .92rem;}
.status-card {
    padding: .8rem 1rem;
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 14px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_state():
    defaults = {
        "file_hash": None,
        "file_name": None,
        "text": None,
        "pages": 0,
        "chunks": None,
        "index": None,
        "quiz": None,
        "quiz_submitted": False,
        "quiz_score": None,
        "weak_topics": [],
        "summary": None,
        "topics": None,
        "flashcards": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_document_outputs():
    st.session_state.quiz = None
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = None
    st.session_state.weak_topics = []
    st.session_state.summary = None
    st.session_state.topics = None
    st.session_state.flashcards = None


def context_for_general_task(text, max_chars=18000):
    if len(text) <= max_chars:
        return text
    # Take material from beginning + middle + end to reduce first-pages bias.
    third = max_chars // 3
    mid = len(text) // 2
    return (
        text[:third]
        + "\n\n[...middle excerpt...]\n\n"
        + text[max(0, mid - third // 2): mid + third // 2]
        + "\n\n[...final excerpt...]\n\n"
        + text[-third:]
    )


init_state()

with st.sidebar:
    st.title("🎓 StudyMate AI")
    st.caption("Upload • Understand • Practice • Improve")

    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL)
    base_url = st.text_input("Ollama URL", value=DEFAULT_BASE_URL)

    online = is_ollama_running(base_url)
    if online:
        st.success("Local AI: Connected")
    else:
        st.warning("Local AI: Not connected")

    st.divider()
    st.markdown("**Competition modules**")
    st.markdown(
        "✅ PDF RAG Q&A  \n"
        "✅ Summary  \n"
        "✅ Important Topics  \n"
        "✅ Quiz + Score  \n"
        "✅ Flashcards  \n"
        "✅ Study Plan"
    )

st.markdown(
    """
<div class="hero">
<h1 style="margin-bottom:.2rem;">🎓 StudyMate AI</h1>
<h3 style="margin-top:0;">Intelligent Personal Learning Assistant</h3>
<p class="small-muted">A local-first RAG system that turns study PDFs into answers, revision material, quizzes and personalized study support.</p>
</div>
""",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload your study material (PDF)",
    type=["pdf"],
    help="Use a text-based PDF for best results.",
)

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    current_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.file_hash != current_hash:
        try:
            with st.spinner("Reading PDF and building semantic search index..."):
                text, pages = extract_pdf(uploaded)
                if not text.strip():
                    st.error(
                        "No selectable text was found in this PDF. "
                        "Try a text-based PDF instead of a scanned-image PDF."
                    )
                    st.stop()

                chunks = split_text(text)
                index = build_index(chunks)

                st.session_state.file_hash = current_hash
                st.session_state.file_name = uploaded.name
                st.session_state.text = text
                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.index = index
                reset_document_outputs()
        except Exception as e:
            st.error(f"Could not process the PDF: {e}")
            st.stop()

if not st.session_state.text:
    st.info("👆 Upload a PDF to start StudyMate AI.")
    st.markdown(
        """
### What happens after upload?
1. PDF text is extracted.
2. Text is split into chunks.
3. Sentence embeddings are created.
4. FAISS builds a semantic-search index.
5. Relevant chunks are sent to the local LLM for grounded answers.
"""
    )
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Document", st.session_state.file_name)
col2.metric("Pages", st.session_state.pages)
col3.metric("RAG Chunks", len(st.session_state.chunks))
col4.metric("Local AI", "Ready" if online else "Offline")

tabs = st.tabs(
    [
        "💬 Ask AI",
        "📖 Summary",
        "🎯 Important Topics",
        "📝 Quiz",
        "🧠 Flashcards",
        "📅 Study Plan",
        "🧩 How it works",
    ]
)

with tabs[0]:
    st.subheader("Ask questions from your PDF")
    level = st.selectbox(
        "Explanation level",
        [
            "Beginner",
            "Engineering Student",
            "2-Mark Exam Answer",
            "5-Mark Exam Answer",
            "10-Mark Exam Answer",
        ],
    )
    question = st.text_input(
        "Your question",
        placeholder="Example: Explain Maxwell's equations in simple words.",
    )

    if st.button("Ask StudyMate", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif not online:
            st.error("Start Ollama first, then click Ask StudyMate again.")
        else:
            try:
                with st.spinner("Retrieving relevant material and generating answer..."):
                    matches = retrieve(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        k=5,
                    )
                    context = "\n\n".join(m["text"] for m in matches)
                    answer = answer_question(
                        question, context, level, model_name, base_url
                    )

                st.markdown("### 🤖 Answer")
                st.markdown(answer)

                with st.expander("See retrieved PDF evidence"):
                    for i, match in enumerate(matches, 1):
                        st.markdown(
                            f"**Match {i} — similarity {match['score']:.3f}**"
                        )
                        st.write(match["text"])
                        st.divider()
            except OllamaError as e:
                st.error(str(e))

with tabs[1]:
    st.subheader("Smart Revision Summary")
    if st.button("Generate Summary"):
        if not online:
            st.error("Start Ollama first.")
        else:
            try:
                with st.spinner("Generating summary..."):
                    st.session_state.summary = summarize_material(
                        context_for_general_task(st.session_state.text),
                        model_name,
                        base_url,
                    )
            except OllamaError as e:
                st.error(str(e))

    if st.session_state.summary:
        st.markdown(st.session_state.summary)

with tabs[2]:
    st.subheader("Important Topic Analysis")
    st.caption("This ranks topics from the uploaded material; it does not guarantee exam questions.")

    if st.button("Analyze Important Topics"):
        if not online:
            st.error("Start Ollama first.")
        else:
            try:
                with st.spinner("Analyzing topics..."):
                    st.session_state.topics = extract_topics(
                        context_for_general_task(st.session_state.text),
                        model_name,
                        base_url,
                    )
            except OllamaError as e:
                st.error(str(e))

    if st.session_state.topics:
        st.markdown(st.session_state.topics)

with tabs[3]:
    st.subheader("AI Quiz")
    quiz_count = st.slider("Number of questions", 3, 8, 5)

    if st.button("Generate New Quiz"):
        if not online:
            st.error("Start Ollama first.")
        else:
            try:
                with st.spinner("Creating quiz..."):
                    quiz = make_quiz(
                        context_for_general_task(st.session_state.text, 14000),
                        model_name,
                        base_url,
                        quiz_count,
                    )
                if not quiz:
                    st.error("No quiz questions were generated. Try again.")
                else:
                    st.session_state.quiz = quiz
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = None
                    st.session_state.weak_topics = []
            except OllamaError as e:
                st.error(str(e))

    if st.session_state.quiz:
        answers = []
        with st.form("quiz_form"):
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**Q{i+1}. {q.get('question','')}**")
                opts = q.get("options", ["A", "B", "C", "D"])
                choice = st.radio(
                    "Choose one",
                    list(range(len(opts))),
                    format_func=lambda x, opts=opts: opts[x],
                    key=f"quiz_q_{i}",
                    index=None,
                )
                answers.append(choice)
                st.divider()

            submitted = st.form_submit_button("Submit Quiz", type="primary")

        if submitted:
            correct = 0
            weak = []
            for i, (q, selected) in enumerate(zip(st.session_state.quiz, answers)):
                ans = q.get("answer_index", -1)
                if selected == ans:
                    correct += 1
                else:
                    weak.append(q.get("question", f"Question {i+1}"))

            total = len(st.session_state.quiz)
            st.session_state.quiz_score = f"{correct}/{total} ({round(correct/total*100)}%)"
            st.session_state.weak_topics = weak
            st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:
            st.success(f"Score: {st.session_state.quiz_score}")
            for i, q in enumerate(st.session_state.quiz):
                ans = q.get("answer_index", 0)
                options = q.get("options", [])
                if 0 <= ans < len(options):
                    with st.expander(f"Q{i+1} answer"):
                        st.write(f"**Correct answer:** {options[ans]}")
                        st.write(q.get("explanation", ""))

with tabs[4]:
    st.subheader("Revision Flashcards")
    if st.button("Generate Flashcards"):
        if not online:
            st.error("Start Ollama first.")
        else:
            try:
                with st.spinner("Creating flashcards..."):
                    st.session_state.flashcards = make_flashcards(
                        context_for_general_task(st.session_state.text, 14000),
                        model_name,
                        base_url,
                    )
            except OllamaError as e:
                st.error(str(e))

    if st.session_state.flashcards:
        st.markdown(st.session_state.flashcards)

with tabs[5]:
    st.subheader("Personalized Study Plan")
    st.caption("Best results come after taking the quiz at least once.")

    if st.button("Create Study Plan"):
        if not online:
            st.error("Start Ollama first.")
        else:
            weak = "\n".join(st.session_state.weak_topics[:5])
            score = st.session_state.quiz_score or "No quiz attempted"
            topic_context = (
                st.session_state.topics
                or context_for_general_task(st.session_state.text, 7000)
            )
            try:
                with st.spinner("Preparing study plan..."):
                    plan = create_study_plan(
                        topic_context,
                        score,
                        weak,
                        model_name,
                        base_url,
                    )
                st.markdown(plan)
            except OllamaError as e:
                st.error(str(e))

with tabs[6]:
    st.subheader("RAG Architecture")
    st.code(
        """
PDF Upload
   ↓
PyMuPDF Text Extraction
   ↓
Chunking
   ↓
SentenceTransformer Embeddings
   ↓
FAISS Semantic Search
   ↓
Top Relevant PDF Chunks
   ↓
Local Ollama LLM
   ↓
Grounded Answer / Summary / Quiz / Flashcards
        """,
        language="text",
    )
    st.markdown(
        """
**Why this is AI/ML:** the embedding model converts text into semantic vectors.
FAISS then finds material that is closest in meaning to the student's question.
The retrieved material is supplied to the local language model, producing a
Retrieval-Augmented Generation (RAG) workflow.
"""
    )
