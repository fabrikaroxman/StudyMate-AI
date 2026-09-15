\
from .ollama_client import generate, generate_json


def answer_question(question, context, level, model, base_url):
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
    return generate(prompt, model=model, base_url=base_url, temperature=0.2)


def summarize_material(text, model, base_url):
    prompt = f"""
You are StudyMate AI. Summarize the study material below for revision.

Use these sections:
## Overview
## Key Concepts
## Important Definitions
## Important Formulas (only if present)
## Exam Revision Points

Be concise but useful. Do not add facts not present in the material.

MATERIAL:
{text}
"""
    return generate(prompt, model=model, base_url=base_url, temperature=0.2)


def extract_topics(text, model, base_url):
    prompt = f"""
Analyze the study material and identify the 10 most important topics.

Return clear Markdown with:
- Topic name
- Priority: High / Medium / Low
- One-line reason

Do not claim exam certainty. Base importance only on the supplied material.

MATERIAL:
{text}
"""
    return generate(prompt, model=model, base_url=base_url, temperature=0.2)


def make_flashcards(text, model, base_url):
    prompt = f"""
Create 10 revision flashcards from this study material.

Format each as:
### Card N
**Q:** ...
**A:** ...

Keep answers short and factual. Use only the supplied material.

MATERIAL:
{text}
"""
    return generate(prompt, model=model, base_url=base_url, temperature=0.2)


def make_quiz(text, model, base_url, count=5):
    prompt = f"""
Create exactly {count} multiple-choice questions from the supplied study material.

Return JSON with this exact structure:
{{
  "questions": [
    {{
      "question": "question text",
      "options": ["option A", "option B", "option C", "option D"],
      "answer_index": 0,
      "explanation": "short explanation"
    }}
  ]
}}

Rules:
- Exactly four options per question.
- answer_index must be 0, 1, 2, or 3.
- Use only facts in the supplied material.
- Avoid ambiguous questions.

MATERIAL:
{text}
"""
    data = generate_json(prompt, model=model, base_url=base_url)
    questions = data.get("questions", [])
    return questions[:count]


def create_study_plan(topic_text, quiz_score, weak_topics, model, base_url):
    prompt = f"""
Create a practical 1-day revision plan for a student.

Available material/topics:
{topic_text}

Quiz score: {quiz_score}
Weak topics:
{weak_topics or "Not enough quiz data yet"}

Return Markdown with:
## Priority Topics
## Study Blocks
## Practice Plan
## Final Revision Checklist

Do not make medical, sleep-deprivation, or extreme-study recommendations.
Keep it realistic and student-friendly.
"""
    return generate(prompt, model=model, base_url=base_url, temperature=0.2)
