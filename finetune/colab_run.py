"""Run Gemma_finetune_on_Colab.ipynb inside an ALREADY CONNECTED Colab session (e.g. the Gemma serving
notebook's runtime), without opening the notebook itself. In a Colab cell:

    !git clone -q --depth 1 -b gemma https://github.com/thenushans-magicktech/qwen2.5.git /content/ft
    exec(open("/content/ft/finetune/colab_run.py").read())

Settings can be changed before the exec line, e.g. OVERRIDES = {"EPOCHS": 2, "SAVE_TO_DRIVE": True}.
Default here: no Google Drive (it needs a sign-in prompt) - download the .gguf from the Files panel instead.
"""
import json
import os

from IPython import get_ipython

_ip = get_ipython()
_nb = json.load(open("/content/ft/finetune/Gemma_finetune_on_Colab.ipynb"))
_cells = ["".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code"]
_over = dict({"SAVE_TO_DRIVE": False}, **globals().get("OVERRIDES", {}))
os.chdir("/content")
for _i, _src in enumerate(_cells):
    print(f"\n===== fine-tune notebook: code cell {_i + 1} of {len(_cells)} =====", flush=True)
    _r = _ip.run_cell(_src)
    if _i == 0:  # the settings cell: apply overrides
        _ip.run_cell("\n".join(f"{k} = {v!r}" for k, v in _over.items()))
    if not _r.success:
        raise SystemExit(f"Stopped: code cell {_i + 1} failed (see the error above).")
print("\nAll done. The .gguf is in /content/gguf_out - download it from the Files panel (left), "
      "or run: from google.colab import files; files.download(GGUF)")
