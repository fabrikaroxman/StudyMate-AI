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


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

model_name = DEFAULT_MODEL
base_url = DEFAULT_BASE_URL
online = is_ollama_running(base_url)


# --------------------------------------------------
# CUSTOM UI
# --------------------------------------------------

CUSTOM_CSS = """
<style>

/* Main container */
.block-container {
    max-width: 1300px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hide sidebar completely */
[data-testid="stSidebar"] {
    display: none;
}

[data-testid="collapsedControl"] {
    display: none;
}

/* Hero */
.hero {
    padding: 2rem 2.2rem;
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 22px;
    margin-bottom: 1.5rem;
    background:
        linear-gradient(
            135deg,
            rgba(59,130,246,.10),
            rgba(139,92,246,.06)
        );
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
}

.hero-subtitle {
    font-size: 1.55rem;
    font-weight: 650;
    margin-top: .7rem;
}

.hero-description {
    opacity: .72;
    font-size: 1rem;
    margin-top: .8rem;
}

/* Metric cards */
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.20);
    border-radius: 16px;
    padding: 1rem;
    background: rgba(128,128,128,.04);
}

[data-testid="stMetricValue"] {
    font-size: 1.8rem;
}

/* Upload box */
[data-testid="stFileUploader"] {
    padding-top: .3rem;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    border-radius: 10px;
    font-weight: 600;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
}

/* Small text */
.small-muted {
    opacity: .72;
}

/* Status */
.cloud-ready {
    display: inline-block;
    padding: .35rem .7rem;
    border-radius: 20px;
    background: rgba(34,197,94,.12);
    font-weight: 600;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

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

    third = max_chars // 3
    mid = len(text) // 2

    return (
        text[:third]
        + "\n\n[...middle excerpt...]\n\n"
        + text[
            max(0, mid - third // 2):
            mid + third // 2
        ]
        + "\n\n[...final excerpt...]\n\n"
        + text[-third:]
    )


init_state()


# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🎓 StudyMate AI
        </div>

        <div class="hero-subtitle">
            Intelligent Personal Learning Assistant
        </div>

        <div class="hero-description">
            Upload your study PDF and transform it into
            answers, summaries, quizzes, flashcards and
            personalized study plans with AI.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded = st.file_uploader(
    "📄 Upload your study material (PDF)",
    type=["pdf"],
    help="Upload a text-based PDF for best results.",
)


if uploaded is not None:

    file_bytes = uploaded.getvalue()
    current_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.file_hash != current_hash:

        try:

            with st.spinner(
                "Reading PDF and building AI search index..."
            ):

                text, pages = extract_pdf(uploaded)

                if not text.strip():

                    st.error(
                        "No selectable text was found in this PDF. "
                        "Please upload a text-based PDF."
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

            st.error(
                f"Could not process the PDF: {e}"
            )

            st.stop()


# --------------------------------------------------
# NO PDF
# --------------------------------------------------

if not st.session_state.text:

    st.info(
        "👆 Upload a PDF to start learning with StudyMate AI."
    )

    st.markdown(
        """
        ### ✨ What can StudyMate AI do?

        **💬 Ask AI** — Ask questions directly from your PDF.

        **📖 Smart Summary** — Generate revision notes.

        **🎯 Important Topics** — Identify important concepts.

        **📝 AI Quiz** — Test your knowledge.

        **🧠 Flashcards** — Quick revision cards.

        **📅 Study Plan** — Generate a personalized revision plan.
        """
    )

    st.stop()


# --------------------------------------------------
# DOCUMENT INFORMATION
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "📄 Document",
    st.session_state.file_name,
)

col2.metric(
    "📚 Pages",
    st.session_state.pages,
)

col3.metric(
    "🧩 RAG Chunks",
    len(st.session_state.chunks),
)

col4.metric(
    "☁️ Cloud AI",
    "Ready" if online else "Unavailable",
)


st.markdown("---")


# --------------------------------------------------
# TABS
# --------------------------------------------------

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


# --------------------------------------------------
# ASK AI
# --------------------------------------------------

with tabs[0]:

    st.subheader(
        "💬 Ask questions from your PDF"
    )

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
        placeholder=(
            "Example: Explain this concept in simple words."
        ),
    )

    if st.button(
        "✨ Ask StudyMate",
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        elif not online:

            st.error(
                "Cloud AI is currently unavailable."
            )

        else:

            try:

                with st.spinner(
                    "Finding relevant information..."
                ):

                    matches = retrieve(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        k=5,
                    )

                    context = "\n\n".join(
                        m["text"]
                        for m in matches
                    )

                    answer = answer_question(
                        question,
                        context,
                        level,
                        model_name,
                        base_url,
                    )

                st.markdown("### 🤖 StudyMate Answer")

                st.markdown(answer)

                with st.expander(
                    "📚 View retrieved PDF evidence"
                ):

                    for i, match in enumerate(
                        matches,
                        1,
                    ):

                        st.markdown(
                            f"**Match {i} — "
                            f"similarity "
                            f"{match['score']:.3f}**"
                        )

                        st.write(
                            match["text"]
                        )

                        st.divider()

            except OllamaError as e:

                st.error(str(e))


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

with tabs[1]:

    st.subheader(
        "📖 Smart Revision Summary"
    )

    st.caption(
        "Generate concise revision notes from your PDF."
    )

    if st.button(
        "✨ Generate Summary"
    ):

        if not online:

            st.error(
                "Cloud AI is currently unavailable."
            )

        else:

            try:

                with st.spinner(
                    "Creating your revision summary..."
                ):

                    st.session_state.summary = (
                        summarize_material(
                            context_for_general_task(
                                st.session_state.text
                            ),
                            model_name,
                            base_url,
                        )
                    )

            except OllamaError as e:

                st.error(str(e))

    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )


# --------------------------------------------------
# IMPORTANT TOPICS
# --------------------------------------------------

with tabs[2]:

    st.subheader(
        "🎯 Important Topic Analysis"
    )

    st.caption(
        "Identify important concepts from the uploaded "
        "material. This does not guarantee exam questions."
    )

    if st.button(
        "🔍 Analyze Important Topics"
    ):

        if not online:

            st.error(
                "Cloud AI is currently unavailable."
            )

        else:

            try:

                with st.spinner(
                    "Analyzing important topics..."
                ):

                    st.session_state.topics = (
                        extract_topics(
                            context_for_general_task(
                                st.session_state.text
                            ),
                            model_name,
                            base_url,
                        )
                    )

            except OllamaError as e:

                st.error(str(e))

    if st.session_state.topics:

        st.markdown(
            st.session_state.topics
        )


# --------------------------------------------------
# QUIZ
# --------------------------------------------------

with tabs[3]:

    st.subheader(
        "📝 AI Quiz"
    )

    quiz_count = st.slider(
        "Number of questions",
        3,
        8,
        5,
    )

    if st.button(
        "🎯 Generate New Quiz"
    ):

        if not online:

            st.error(
                "Cloud AI is currently unavailable."
            )

        else:

            try:

                with st.spinner(
                    "Creating your quiz..."
                ):

                    quiz = make_quiz(
                        context_for_general_task(
                            st.session_state.text,
                            14000,
                        ),
                        model_name,
                        base_url,
                        quiz_count,
                    )

                if not quiz:

                    st.error(
                        "No quiz questions were generated. "
                        "Please try again."
                    )

                else:

                    st.session_state.quiz = quiz
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = None
                    st.session_state.weak_topics = []

            except OllamaError as e:

                st.error(str(e))

    if st.session_state.quiz:

        answers = []

        with st.form(
            "quiz_form"
        ):

            for i, q in enumerate(
                st.session_state.quiz
            ):

                st.markdown(
                    f"### Q{i + 1}. "
                    f"{q.get('question', '')}"
                )

                opts = q.get(
                    "options",
                    [
                        "A",
                        "B",
                        "C",
                        "D",
                    ],
                )

                choice = st.radio(
                    "Choose one",
                    list(range(len(opts))),
                    format_func=lambda x, opts=opts: (
                        opts[x]
                    ),
                    key=f"quiz_q_{i}",
                    index=None,
                )

                answers.append(choice)

                st.divider()

            submitted = st.form_submit_button(
                "✅ Submit Quiz",
                type="primary",
            )

        if submitted:

            correct = 0
            weak = []

            for i, (
                q,
                selected,
            ) in enumerate(
                zip(
                    st.session_state.quiz,
                    answers,
                )
            ):

                ans = q.get(
                    "answer_index",
                    -1,
                )

                if selected == ans:

                    correct += 1

                else:

                    weak.append(
                        q.get(
                            "question",
                            f"Question {i + 1}",
                        )
                    )

            total = len(
                st.session_state.quiz
            )

            percentage = round(
                correct / total * 100
            )

            st.session_state.quiz_score = (
                f"{correct}/{total} "
                f"({percentage}%)"
            )

            st.session_state.weak_topics = weak
            st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:

            st.success(
                "🏆 Score: "
                + st.session_state.quiz_score
            )

            for i, q in enumerate(
                st.session_state.quiz
            ):

                ans = q.get(
                    "answer_index",
                    0,
                )

                options = q.get(
                    "options",
                    [],
                )

                if 0 <= ans < len(options):

                    with st.expander(
                        f"Q{i + 1} Answer"
                    ):

                        st.write(
                            "**Correct answer:** "
                            + options[ans]
                        )

                        st.write(
                            q.get(
                                "explanation",
                                "",
                            )
                        )


# --------------------------------------------------
# FLASHCARDS
# --------------------------------------------------

with tabs[4]:

    st.subheader(
        "🧠 Revision Flashcards"
    )

    if st.button(
        "✨ Generate Flashcards"
    ):

        if not online:

            st.error(
                "Cloud AI is currently unavailable."
            )

        else:

            try:

                with st.spinner(
                    "Creating flashcards..."
                ):

                    st.session_state.flashcards = (
                        make_flashcards(
                            context_for_general_task(
                                st.session_state.text,
                                14000,
                            ),
                            model_name,
                            base_url,
                        )
                    )

            except OllamaError as e:

                st.error(str(e))

    if st.session_state.flashcards:

        st.markdown(
            st.session_state.flashcards
        )


# --------------------------------------------------
# STUDY PLAN
# --------------------------------------------------

with tabs[5]:

    st.subheader(
        "📅 Personalized Study Plan"
    )

    st.caption(
        "For best results, take the AI Quiz first."
    )

    if st.button(
        "🚀 Create Study Plan"
    ):

        if not online:

            st.error(
                "Cloud AI is currently unavailable."
            )

        else:

            weak = "\n".join(
                st.session_state.weak_topics[:5]
            )

            score = (
                st.session_state.quiz_score
                or "No quiz attempted"
            )

            topic_context = (
                st.session_state.topics
                or context_for_general_task(
                    st.session_state.text,
                    7000,
                )
            )

            try:

                with st.spinner(
                    "Preparing your study plan..."
                ):

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


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

with tabs[6]:

    st.subheader(
        "🧩 How StudyMate AI Works"
    )

    st.code(
        """
PDF Upload
    ↓
Text Extraction
    ↓
Smart Chunking
    ↓
SentenceTransformer Embeddings
    ↓
FAISS Vector Search
    ↓
Relevant PDF Content
    ↓
Groq Cloud AI
    ↓
Answer / Summary / Quiz / Flashcards / Study Plan
        """,
        language="text",
    )

    st.markdown(
        """
### 🤖 AI / ML Architecture

StudyMate AI uses **semantic embeddings** to convert
study material into numerical vectors.

**FAISS Vector Search** finds the PDF content that is
most relevant to the student's question.

The relevant information is then supplied to
**Groq Cloud AI** to generate a grounded response.

This creates a **Retrieval-Augmented Generation
(RAG)** learning system.
        """
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        opacity:.65;
        padding:1rem;
    ">
        🎓 <b>StudyMate AI</b><br>
        Upload • Understand • Practice • Improve
    </div>
    """,
    unsafe_allow_html=True,
)