from .ollama_client import generate, generate_json


# Maximum characters sent in one AI request.
# This keeps large PDFs from overflowing API/rate limits.
MAX_CHARS = 12000


def _split_text(text, max_chars=MAX_CHARS):
    """Split very large study material into safe batches."""
    text = text or ""

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        # Try to break at a paragraph boundary.
        if end < len(text):
            break_at = text.rfind("\n\n", start, end)

            if break_at > start + (max_chars // 2):
                end = break_at

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end

    return chunks


def _sample_large_text(text, max_batches=6):
    """
    Select representative sections from a very large PDF.
    This avoids sending 100,000+ words to the AI at once.
    """
    chunks = _split_text(text)

    if len(chunks) <= max_batches:
        return chunks

    # Pick sections distributed throughout the document.
    indexes = []

    for i in range(max_batches):
        index = round(i * (len(chunks) - 1) / (max_batches - 1))

        if index not in indexes:
            indexes.append(index)

    return [chunks[i] for i in indexes]


def answer_question(question, context, level, model, base_url):
    # RAG already supplies relevant context here,
    # so keep only a safe amount.
    context = (context or "")[:MAX_CHARS]

    prompt = f"""
You are StudyMate AI, an academic learning assistant.

Rules:
1. Answer ONLY from the supplied study material.
2. If the answer is not supported by the material, say:
   "I could not find enough information in the uploaded material."
3. Do not invent facts.
4. Keep the answer clear and useful for a student.
5. Explanation level: {level}

STUDY MATERIAL:
{context}

QUESTION:
{question}

Give the final answer now.
"""

    return generate(
        prompt,
        model=model,
        base_url=base_url,
        temperature=0.2,
    )


def summarize_material(text, model, base_url):
    batches = _sample_large_text(text, max_batches=6)

    partial_summaries = []

    for number, batch in enumerate(batches, start=1):
        prompt = f"""
You are StudyMate AI.

Summarize this section of a large study document.

Focus on:
- Main concepts
- Important definitions
- Important formulas if present
- Important revision points

Do not add information that is not present.

DOCUMENT SECTION {number}/{len(batches)}:

{batch}
"""

        result = generate(
            prompt,
            model=model,
            base_url=base_url,
            temperature=0.2,
        )

        partial_summaries.append(result)

    combined = "\n\n".join(partial_summaries)

    # Keep final synthesis request controlled too.
    combined = combined[:MAX_CHARS]

    final_prompt = f"""
You are StudyMate AI.

Combine the section summaries below into one useful revision summary.

Use exactly these sections:

## Overview
## Key Concepts
## Important Definitions
## Important Formulas
## Exam Revision Points

Remove repetition.
Do not invent facts.
Keep the final summary clear and concise.

SECTION SUMMARIES:

{combined}
"""

    return generate(
        final_prompt,
        model=model,
        base_url=base_url,
        temperature=0.2,
    )


def extract_topics(text, model, base_url):
    batches = _sample_large_text(text, max_batches=5)

    topic_results = []

    for batch in batches:
        prompt = f"""
Analyze this section of study material.

Identify up to 8 important topics.

For each topic provide:
- Topic name
- Importance reason

Do not predict exam questions.
Use only the supplied material.

MATERIAL:

{batch}
"""

        result = generate(
            prompt,
            model=model,
            base_url=base_url,
            temperature=0.2,
        )

        topic_results.append(result)

    combined = "\n\n".join(topic_results)
    combined = combined[:MAX_CHARS]

    final_prompt = f"""
From the topic analyses below, identify the 10 most important
topics across the document.

Return Markdown using:

### Topic Name
**Priority:** High / Medium / Low
**Reason:** one short reason

Remove duplicate topics.

Do not claim exam certainty.

TOPIC ANALYSES:

{combined}
"""

    return generate(
        final_prompt,
        model=model,
        base_url=base_url,
        temperature=0.2,
    )


def make_flashcards(text, model, base_url):
    batches = _sample_large_text(text, max_batches=4)

    material = "\n\n".join(batches)
    material = material[:MAX_CHARS]

    prompt = f"""
Create 10 revision flashcards from the supplied study material.

Format:

### Card 1
**Q:** Question
**A:** Answer

Continue until Card 10.

Rules:
- Keep answers short.
- Use only supplied material.
- Cover different important concepts.
- Avoid duplicate questions.

MATERIAL:

{material}
"""

    return generate(
        prompt,
        model=model,
        base_url=base_url,
        temperature=0.2,
    )


def make_quiz(text, model, base_url, count=5):
    batches = _sample_large_text(text, max_batches=4)

    material = "\n\n".join(batches)
    material = material[:MAX_CHARS]

    prompt = f"""
Create exactly {count} multiple-choice questions
from the supplied study material.

Return JSON with this exact structure:

{{
  "questions": [
    {{
      "question": "question text",
      "options": [
        "option A",
        "option B",
        "option C",
        "option D"
      ],
      "answer_index": 0,
      "explanation": "short explanation"
    }}
  ]
}}

Rules:
- Exactly four options per question.
- answer_index must be 0, 1, 2, or 3.
- Use only facts from the supplied material.
- Avoid ambiguous questions.
- Cover different concepts.

MATERIAL:

{material}
"""

    data = generate_json(
        prompt,
        model=model,
        base_url=base_url,
    )

    questions = data.get("questions", [])

    return questions[:count]


def create_study_plan(
    topic_text,
    quiz_score,
    weak_topics,
    model,
    base_url,
):
    topic_text = (topic_text or "")[:MAX_CHARS]

    prompt = f"""
Create a practical 1-day revision plan for a student.

Available material/topics:

{topic_text}

Quiz score:
{quiz_score}

Weak topics:
{weak_topics or "Not enough quiz data yet"}

Return Markdown with:

## Priority Topics
## Study Blocks
## Practice Plan
## Final Revision Checklist

Keep the plan realistic and student-friendly.
Do not recommend extreme study schedules or sleep deprivation.
"""

    return generate(
        prompt,
        model=model,
        base_url=base_url,
        temperature=0.2,
    )