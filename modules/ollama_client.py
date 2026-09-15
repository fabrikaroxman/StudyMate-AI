\
import json
import requests


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"


class OllamaError(RuntimeError):
    pass


def _post(path, payload, base_url=DEFAULT_BASE_URL, timeout=180):
    url = base_url.rstrip("/") + path
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            "Ollama is not running. Start Ollama, then try again."
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaError(
            "The local model took too long to respond. Try a smaller model."
        ) from e
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = r.json().get("error", "")
        except Exception:
            pass
        raise OllamaError(detail or f"Ollama returned HTTP {r.status_code}.") from e


def is_ollama_running(base_url=DEFAULT_BASE_URL):
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=4)
        return r.ok
    except requests.RequestException:
        return False


def generate(prompt, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL,
             temperature=0.2, num_ctx=8192):
    data = _post(
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
        base_url=base_url,
    )
    return data.get("response", "").strip()


def generate_json(prompt, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL):
    data = _post(
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
        base_url=base_url,
    )
    raw = data.get("response", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise OllamaError("The model returned invalid JSON. Please try again.") from e
