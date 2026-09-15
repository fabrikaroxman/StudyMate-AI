import json
import os
import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"
GROQ_MODEL = "openai/gpt-oss-20b"


class OllamaError(RuntimeError):
    pass


def _get_groq_key():
    """Read Groq API key from Streamlit Secrets or environment."""
    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


def using_cloud_ai():
    return bool(_get_groq_key())


def _groq_generate(prompt, temperature=0.2, json_mode=False):
    api_key = _get_groq_key()

    if not api_key:
        raise OllamaError("GROQ_API_KEY is not configured.")

    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are StudyMate AI, an academic learning assistant. "
                    "Follow the requested output format carefully."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout as exc:
        raise OllamaError(
            "Cloud AI took too long to respond."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            "Could not connect to the cloud AI service."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        try:
            message = response.json()["error"]["message"]
        except Exception:
            message = f"Cloud AI returned HTTP {response.status_code}."

        raise OllamaError(message) from exc

    except (KeyError, IndexError, TypeError) as exc:
        raise OllamaError(
            "Cloud AI returned an unexpected response."
        ) from exc


def _ollama_post(path, payload, base_url, timeout=180):
    url = base_url.rstrip("/") + path

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            "Ollama is not running. Start Ollama and try again."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise OllamaError(
            "The local model took too long to respond."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        try:
            message = response.json().get("error", "")
        except Exception:
            message = ""

        raise OllamaError(
            message or f"Ollama returned HTTP {response.status_code}."
        ) from exc


def is_ollama_running(base_url=DEFAULT_BASE_URL):
    # On Streamlit Cloud, Groq becomes the AI backend.
    if using_cloud_ai():
        return True

    try:
        response = requests.get(
            base_url.rstrip("/") + "/api/tags",
            timeout=4,
        )
        return response.ok

    except requests.RequestException:
        return False


def generate(
    prompt,
    model=DEFAULT_MODEL,
    base_url=DEFAULT_BASE_URL,
    temperature=0.2,
    num_ctx=8192,
):
    if using_cloud_ai():
        return _groq_generate(
            prompt,
            temperature=temperature,
            json_mode=False,
        )

    data = _ollama_post(
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        },
        base_url,
    )

    return data.get("response", "").strip()


def generate_json(
    prompt,
    model=DEFAULT_MODEL,
    base_url=DEFAULT_BASE_URL,
):
    if using_cloud_ai():
        raw = _groq_generate(
            prompt,
            temperature=0.1,
            json_mode=True,
        )

    else:
        data = _ollama_post(
            "/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 8192,
                },
            },
            base_url,
        )

        raw = data.get("response", "").strip()

    try:
        return json.loads(raw)

    except json.JSONDecodeError as exc:
        raise OllamaError(
            "The AI returned invalid JSON. Please try again."
        ) from exc