import pytest

from modules import features
from modules.features import (
    _clean_questions,
    _fit_batches,
    _to_answer_index,
    make_quiz,
)


# ---------- _fit_batches ----------

def test_fit_batches_keeps_short_input_unchanged():
    assert _fit_batches(["a", "b"], limit=100) == "a\n\nb"


def test_fit_batches_ignores_empty_and_none():
    assert _fit_batches(None) == ""
    assert _fit_batches(["", ""]) == ""


def test_fit_batches_represents_every_batch():
    batches = ["A" * 5000, "B" * 5000, "C" * 5000, "D" * 5000]
    out = _fit_batches(batches, limit=12000)
    assert len(out) <= 12000
    # The old join(...)[:limit] would have dropped D (and most of C).
    for marker in "ABCD":
        assert marker in out


# ---------- _to_answer_index ----------

@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        (3, 3),
        ("2", 2),
        (" 1 ", 1),
        (1.0, 1),
        ("B", 1),
        ("d", 3),
        ("Paris", 1),
        (4, None),
        (-1, None),
        ("7", None),
        ("Berlin", None),
        (None, None),
        (True, None),
        ([1], None),
    ],
)
def test_to_answer_index(value, expected):
    assert _to_answer_index(value, ["Rome", "Paris", "Madrid", "Lisbon"]) == expected


# ---------- _clean_questions ----------

def _q(**over):
    base = {
        "question": "Capital of France?",
        "options": ["Rome", "Paris", "Madrid", "Lisbon"],
        "answer_index": 1,
        "explanation": "Paris is the capital.",
    }
    base.update(over)
    return base


def test_clean_questions_keeps_valid_question():
    out = _clean_questions({"questions": [_q()]}, 5)
    assert out == [_q()]


def test_clean_questions_coerces_string_answer_index():
    out = _clean_questions({"questions": [_q(answer_index="1")]}, 5)
    assert out[0]["answer_index"] == 1
    assert isinstance(out[0]["answer_index"], int)


def test_clean_questions_drops_broken_items():
    raw = {
        "questions": [
            _q(),
            _q(question=""),
            _q(options="not a list"),
            _q(options=["only one"]),
            _q(answer_index=9),
            _q(answer_index=None),
            "garbage",
            None,
        ]
    }
    assert len(_clean_questions(raw, 5)) == 1


def test_clean_questions_respects_count():
    raw = {"questions": [_q(question=f"Q{i}") for i in range(10)]}
    assert len(_clean_questions(raw, 3)) == 3


def test_clean_questions_accepts_bare_list_and_rejects_junk():
    assert len(_clean_questions([_q()], 5)) == 1
    assert _clean_questions("nope", 5) == []
    assert _clean_questions({"questions": "nope"}, 5) == []
    assert _clean_questions({}, 5) == []


def test_clean_questions_fills_missing_explanation():
    item = _q()
    del item["explanation"]
    assert _clean_questions([item], 5)[0]["explanation"] == ""


# ---------- make_quiz ----------

def test_make_quiz_cleans_model_output_and_requests_more_tokens(monkeypatch):
    seen = {}

    def fake_generate_json(prompt, model, base_url, max_tokens):
        seen["max_tokens"] = max_tokens
        return {"questions": [_q(answer_index="2"), _q(answer_index=99)]}

    monkeypatch.setattr(features, "generate_json", fake_generate_json)

    quiz = make_quiz("some study text", "m", "http://x", count=5)

    assert seen["max_tokens"] == features.QUIZ_MAX_TOKENS
    assert len(quiz) == 1
    assert quiz[0]["answer_index"] == 2


def test_make_quiz_prompt_covers_end_of_long_document(monkeypatch):
    seen = {}

    def fake_generate_json(prompt, model, base_url, max_tokens):
        seen["prompt"] = prompt
        return {"questions": []}

    monkeypatch.setattr(features, "generate_json", fake_generate_json)

    text = "\n\n".join(["start-section " * 600, "middle-section " * 600, "end-section " * 600])
    make_quiz(text, "m", "http://x")

    assert "end-section" in seen["prompt"]
