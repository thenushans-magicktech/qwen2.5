"""Connect the local kitchen designer to Qwen or Gemma running in Colab (or any remote Ollama behind the proxy).

    python colab/connect_colab.py https://<random>.trycloudflare.com <password> [port] [--designer qwen|gemma]

Checks that the URL answers and the password is accepted, shows which models Colab has,
then starts the web app on this laptop (default http://127.0.0.1:8000) with every Ollama
call (Qwen and Gemma) going to Colab. Everything else - UI, rules, catalogue, files - stays local.

``--designer gemma`` (use with colab/Gemma_on_Colab_GPU.ipynb) checks that Gemma is on Colab
and makes it the default designer; Qwen stays selectable if Colab has it too.
GEMINI_API_KEY, if set in this shell, is passed on to the app.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    args = sys.argv[1:]
    designer = "qwen"
    if "--designer" in args:
        i = args.index("--designer")
        designer = args[i + 1] if i + 1 < len(args) else ""
        del args[i:i + 2]
    if len(args) < 2 or designer not in ("qwen", "gemma"):
        print(__doc__)
        return 2
    url, key = args[0].strip().rstrip("/"), args[1].strip()
    port = args[2] if len(args) > 2 else "8000"
    if designer == "gemma":
        want = os.environ.get("GEMMA_MODEL", "gemma4:E4B")
    else:
        want = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")

    req = urllib.request.Request(url + "/api/tags", headers={"Authorization": f"Bearer {key}"})
    t = time.time()
    try:
        models = [m["name"] for m in json.load(urllib.request.urlopen(req, timeout=30))["models"]]
    except urllib.error.HTTPError as exc:
        hint = "the password is wrong (copy it again from the last Colab cell)" if exc.code == 401 else exc.reason
        print(f"Colab answered HTTP {exc.code}: {hint}")
        return 1
    except Exception as exc:  # DNS, timeout, tunnel gone
        print(f"Cannot reach {url}: {exc}")
        print("Is the Colab notebook still running? If the session ended, Run all again and use the new URL and password.")
        return 1
    print(f"Connected to Colab in {time.time() - t:.1f} s - models there: {', '.join(models) or 'none'}")
    if want not in models and f"{want}:latest" not in models:
        print(f"Warning: {want} is not on Colab yet - run the 'ollama pull' cell in the notebook.")
        return 1

    env = dict(os.environ, OLLAMA_HOST=url, OLLAMA_API_KEY=key,
               AI_EXPECTED_TIME="under a minute", GEMMA_EXPECTED_TIME="under a minute")
    if designer == "gemma":
        env["MAGICKHOME_AI_PROVIDER"] = "gemma"
    root = Path(__file__).resolve().parents[1]
    name = "Gemma" if designer == "gemma" else "Qwen"
    print(f"Starting the kitchen designer on http://127.0.0.1:{port} ({name} runs on Colab). Press Ctrl+C to stop.")
    return subprocess.call([sys.executable, "main.py", "--serve", "--port", port], cwd=root, env=env)


if __name__ == "__main__":
    sys.exit(main())
