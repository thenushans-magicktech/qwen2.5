"""Connect the local kitchen designer to Qwen running in Colab (or any remote Ollama behind the proxy).

    python colab/connect_colab.py https://<random>.trycloudflare.com <password> [port]

Checks that the URL answers and the password is accepted, shows which models Colab has,
then starts the web app on this laptop (default http://127.0.0.1:8000) with every Qwen
call going to Colab. Everything else - UI, rules, catalogue, files - stays local.
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
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    url, key = sys.argv[1].strip().rstrip("/"), sys.argv[2].strip()
    port = sys.argv[3] if len(sys.argv) > 3 else "8000"
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
    if want not in models:
        print(f"Warning: {want} is not on Colab yet - run the 'ollama pull' cell in the notebook.")
        return 1

    env = dict(os.environ, OLLAMA_HOST=url, OLLAMA_API_KEY=key, AI_EXPECTED_TIME="under a minute")
    root = Path(__file__).resolve().parents[1]
    print(f"Starting the kitchen designer on http://127.0.0.1:{port} (Qwen runs on Colab). Press Ctrl+C to stop.")
    return subprocess.call([sys.executable, "main.py", "--serve", "--port", port], cwd=root, env=env)


if __name__ == "__main__":
    sys.exit(main())
