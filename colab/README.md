# Run only the Qwen model on a GPU (Google Colab), keep everything else local

Only Qwen moves to the GPU. The web UI, rule engine, catalogue, floor plans and generated scenes stay on the
laptop. The laptop sends Qwen the design question and gets the answer back.

```
Laptop: python main.py --serve  --(prompt, password)-->  Colab GPU: proxy -> Ollama -> qwen2.5-coder:7b
```

## Steps
1. Open https://colab.research.google.com → **File → Upload notebook** → choose `colab/Qwen_on_Colab_GPU.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save**, then **Runtime → Run all**. The first run takes 3–5 minutes.
3. The last cell prints the tunnel URL and a password. On your laptop, in the project folder, run:
   ```powershell
   python colab/connect_colab.py https://<random>.trycloudflare.com <password>
   ```
   It checks that Colab answers and accepts the password, then starts the app with every Qwen call going to Colab.
   You can also paste the 4 printed lines instead.
4. Open http://127.0.0.1:8000. The status pill should say "AI designer ready".
5. When you're finished: in Colab, **Runtime → Disconnect and delete runtime**.

Each Colab session has a new URL and password, so repeat steps 2–3 each time.

## Files
- `Qwen_on_Colab_GPU.ipynb`: installs Ollama, downloads Qwen 7B, starts the password proxy and a Cloudflare tunnel,
  then prints the laptop commands.
- `ollama_auth_proxy.py`: the password check, in standard Python with nothing to install. Requests without
  `Authorization: Bearer <password>` get HTTP 401. The notebook runs it on a free port.
  The project's Ollama client sends the header automatically when `OLLAMA_API_KEY` is set.
  The same proxy works on a rented GPU server or an office GPU PC.

## Limits
- **Free Colab** disconnects when idle (about 90 minutes) and after about 12 hours, and a GPU isn't always
  available. Colab is meant for interactive use, so use it for testing.
- **The free Cloudflare tunnel** needs each request to finish within about 100 seconds. A design on a T4 takes
  roughly 20–40 seconds.
- **Privacy:** each design sends the room dimensions, catalogue codes and the brief to Google and through Cloudflare.
