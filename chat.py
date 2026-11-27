"""Design chat: answer the customer's questions about ONE finished design ("why is the sink here?").

The answer is grounded in that design's own report - every decision with the reason the designer gave, the rule-book
rules that were applied, the customer's stated needs and the cabinets on each wall - plus the text of any rule the
question names (R11 ...). That fact sheet goes into the system message, so a small local model only has to explain
what is written there (it never has to remember rules, which fine-tuned models get wrong).

Models, in order (CHAT_MODEL=auto): a llama.cpp server model (fine-tuned 'magick-*' first), then an Ollama model,
then Gemini when GEMINI_API_KEY is set. When no model answers, the matching saved reasons are returned as they are.
"""
from __future__ import annotations

import json
import os
import logging
import re
import time
from typing import Optional

import config

log = logging.getLogger("chat")

SYSTEM = """You are the MagickHome kitchen designer assistant, made by MagickTech. You designed the kitchen described \
in the DESIGN FACTS below. The customer asks about this design.
- Answer only from the DESIGN FACTS (and RULE TEXT when given). Explain the reason that was written for the decision.
- Never invent rules or sizes. Only when NOTHING in the facts is about the question, say you have no note on that.
- Talk to the customer directly: say "you" and "I" (never "the customer", never "the design facts"). Start with the answer itself.
- Be short and friendly: 2-4 sentences, plain words, sizes in mm. No JSON, no headings."""

MAX_HISTORY = 4  # earlier messages sent with a question (2 turns: small models drift with more)


def _wall_names(job: dict) -> dict[str, str]:
    try:
        from scene_parser import parse_scene

        geo = parse_scene(job["result"])
        return {wid: (w.name or wid) for wid, w in geo.walls.items()}
    except Exception:  # an old or odd scene: keep the ids
        return {}


def design_facts(job: dict, placements: bool = True) -> str:
    """The design's fact sheet (about 2-3k tokens)."""
    rep = job.get("report") or {}
    names = _wall_names(job)
    out = []
    rb = rep.get("rulebook") or {}
    if rb:
        out.append(f"Kitchen: {rb.get('kitchen_type', '')}, {rb.get('size_band', '')}, "
                   f"{rb.get('floor_area_sqft', '?')} sqft floor area.")
    needs = rep.get("customer_needs") or []
    if needs:
        out.append("CUSTOMER NEEDS:\n" + "\n".join(needs))
    out.append("DECISIONS (what was built and why):")
    for e in rep.get("explanation") or []:
        line = f"- {e.get('item')}: built {e.get('built')}. Why: {e.get('why')}"
        details = [d for d in (e.get("details") or []) if d][:4]
        if details:
            line += " Details: " + "; ".join(details)
        if e.get("need"):
            line += f" Customer need: {e['need']}"
        out.append(line)
    applied = rb.get("applied") or []
    if applied:
        out.append("RULE-BOOK RULES APPLIED:")
        for a in applied:
            out.append(f"- Rule R{a.get('rule')}: {a.get('text')}" + (f" (designer had proposed {a['designer']})"
                                                                      if a.get("changed") and a.get("designer") else ""))
    lay = rep.get("layout") or {}
    walls = (lay.get("placements") or {}) if placements else {}
    if walls:
        out.append("CABINETS ON EACH WALL (left to right, start position along the wall):")
        for wid, items in walls.items():
            parts = [f"{i.get('category')} {i.get('width')} mm at {round(i.get('start_mm') or 0)} mm ({i.get('tier')})"
                     for i in items]
            out.append(f"- {names.get(wid, wid)}: " + "; ".join(parts))
    for d in (lay.get("decisions") or []) if placements else []:
        out.append(f"- Placement note: {d}")
    for w in lay.get("warnings") or []:
        out.append(f"- Warning: {w}")
    text = "\n".join(out)
    for wid, name in names.items():  # placement notes use wall ids
        text = text.replace(wid, name)
    return text


def rule_texts(question: str, history: list[dict]) -> str:
    nums = {int(n) for n in re.findall(r"\b[Rr](?:ule)?\s*-?\s*(\d{1,3})\b",
                                       " ".join([question] + [m.get("content", "") for m in history[-2:]]))}
    if not nums:
        return ""
    try:
        import rulebook

        rb = rulebook.load()
    except Exception as exc:
        log.warning("rule book not available: %s", exc)
        return ""
    lines = [f"Rule R{n}: {rb.texts[n]}" if rb.texts.get(n) else f"Rule R{n}: not in the rule book" for n in sorted(nums)]
    return "RULE TEXT:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _pick_model() -> Optional[str]:
    want = (config.CHAT_MODEL or "auto").strip()
    if want != "auto":
        return want
    from ai_designer import _llamacpp_servers

    try:
        ids = list(_llamacpp_servers())
    except Exception:
        ids = []
    if ids:
        best = next((m for m in ids if m.startswith("magick")), ids[0])
        return "llamacpp:" + best
    try:
        from ai_designer import _ollama_models

        names = [m["name"] for m in _ollama_models()]
    except Exception:
        names = []
    for pref in ("magick-gemma", config.GEMMA_MODEL, config.LOCAL_MODEL):
        hit = next((n for n in names if n.split(":")[0] == pref.split(":")[0]), None)
        if hit:
            return "ollama:" + hit
    if names:
        return "ollama:" + names[0]
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return "gemini"
    return None


def ask_json(model: str, messages: list[dict], schema: dict) -> dict:
    """Schema-locked JSON answer (llama.cpp / Ollama grammars; Gemini: JSON mode, checked by the caller)."""
    import requests

    if model.startswith("llamacpp:"):
        from ai_designer import _llamacpp_headers, _llamacpp_servers

        mid = model.split(":", 1)[1]
        server = _llamacpp_servers().get(mid)
        if not server:
            raise RuntimeError(f"no llama.cpp server serves {mid}")
        body = {"model": mid, "messages": messages, "temperature": 0, "max_tokens": 700,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "change", "schema": schema}}}
        r = requests.post(f"{server}/v1/chat/completions", json=body, headers=_llamacpp_headers(),
                          timeout=config.CHAT_TIMEOUT_S)
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])
    if model.startswith("ollama:"):
        body = {"model": model.split(":", 1)[1], "messages": messages, "stream": False, "think": False,
                "format": schema, "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 700}}
        r = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=body, timeout=config.CHAT_TIMEOUT_S)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])
    text = _ask(model, messages + [{"role": "user", "content": "Reply with the JSON object only. Schema: "
                                                                 + json.dumps(schema)[:6000]}])
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def _ask(model: str, messages: list[dict]) -> str:
    import requests

    if model.startswith("llamacpp:"):
        from ai_designer import _llamacpp_headers, _llamacpp_servers

        mid = model.split(":", 1)[1]
        server = _llamacpp_servers().get(mid)
        if not server:
            raise RuntimeError(f"no llama.cpp server serves {mid}")
        body = {"model": mid, "messages": messages, "temperature": 0.2, "max_tokens": config.CHAT_MAX_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False}}
        r = requests.post(f"{server}/v1/chat/completions", json=body, headers=_llamacpp_headers(),
                          timeout=config.CHAT_TIMEOUT_S)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if model.startswith("ollama:"):
        body = {"model": model.split(":", 1)[1], "messages": messages, "stream": False, "think": False,
                "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": config.CHAT_MAX_TOKENS}}
        r = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=body, timeout=config.CHAT_TIMEOUT_S)
        r.raise_for_status()
        return r.json()["message"]["content"]
    if model == "gemini":
        sys_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent"
        r = requests.post(url, headers={"x-goog-api-key": os.environ.get("GEMINI_API_KEY", "").strip()},
                          json={"system_instruction": {"parts": [{"text": sys_txt}]}, "contents": contents,
                                "generationConfig": {"temperature": 0.2}}, timeout=config.CHAT_TIMEOUT_S)
        r.raise_for_status()
        return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
    raise RuntimeError(f"unknown chat model {model!r}")


# ---------------------------------------------------------------------------
# No-model answer: the saved reasons that match the question
# ---------------------------------------------------------------------------

_STOP = set("the a an is are was why did you this that here there it of on in to for and or what how do does my "
            "unit units mm wall base cabinet cabinets put place placed".split())


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in _STOP and len(w) > 2}


# a clicked item's type -> the design decision ("explanation" item) that explains it
_ITEM_FOR = (
    (r"\bsink", "Sink"), (r"\bhob\b|chimney|cooktop", "Hob"), (r"cylinder|\blpg", "LPG cylinder"),
    (r"spice", "Spice pull-out"), (r"aquaguard|purifier", "Water purifier"), (r"\bloft", "Lofts"),
    (r"corner|blind", "Corners"), (r"\btall|pantry|oven|microwave|fridge|refrigerator", "Tall units"),
    (r"glass|display", "Glass display"), (r"open", "Open shelf unit"), (r"grain|wicker|basket|trolley", "Feature units"),
    (r"^wall|wall unit|wall cabinet", "Wall cabinets"), (r"^base|drawer|shutter", "Base cabinets"),
)


def _item_for(focus: Optional[dict]) -> Optional[str]:
    title = str((focus or {}).get("title", "")).lower()
    return next((item for pat, item in _ITEM_FOR if re.search(pat, title)), None)


def matching_reasons(job: dict, question: str, focus: Optional[dict] = None, k: int = 2) -> list[dict]:
    q = _words(question)
    target = _item_for(focus)
    if focus:  # a clicked item: match on what it is, not on the question wording ("why is this here?")
        q = _words(f"{focus.get('title', '')} {focus.get('sub', '')}") | q
    scored = []
    for e in (job.get("report") or {}).get("explanation") or []:
        words = _words(f"{e.get('item')} {e.get('built')} {' '.join(e.get('details') or [])}")
        score = 3 * len(q & _words(str(e.get("item")))) + len(q & words)
        if target and str(e.get("item")) == target:  # the clicked item's own decision always comes first
            score += 100
        if focus:  # the clicked item's name ("grain trolley") written in this decision: it is about that item
            text = f"{e.get('built')} {e.get('why')} {' '.join(e.get('details') or [])}".lower()
            name = [w for w in re.findall(r"[a-z]+", str(focus.get("title", "")).lower()) if w not in _STOP]
            pairs = [f"{a} {b}" for a, b in zip(name, name[1:])]
            score += 4 * sum(p in text for p in pairs)
            # a detail line that starts with the item's name is the one about this very unit
            score += 3 * any(d.lower().startswith(p) for d in (e.get("details") or []) for p in pairs)
        if score:
            scored.append((score, e))
    scored.sort(key=lambda t: -t[0])
    return [{"item": e.get("item"), "built": e.get("built"), "why": e.get("why")} for _, e in scored[:k]]


def answer(job: dict, question: str, history: Optional[list[dict]] = None, focus: Optional[dict] = None,
           placements: bool = True) -> dict:
    """``placements=False``: leave out this designer's own cabinet-by-cabinet layout - for apps whose rule engine
    placed the final cabinets differently (exact sizes / positions come from the item the customer clicked)."""
    history = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")][-MAX_HISTORY:]
    if focus:  # a clicked item is a new topic: earlier answers only pull a small model back to them
        history = []
    sources = matching_reasons(job, question, focus)
    system = SYSTEM + "\n\nDESIGN FACTS:\n" + design_facts(job, placements)
    if not placements:
        system += ("\n\nThe exact cabinet sizes and positions in the customer's design come from the rule engine: "
                   "take the clicked item's size and wall as true and explain the DECISION behind it.")  # same for every turn: llama.cpp reuses its cache
    user = question.strip()
    if focus:
        bits = [focus.get("title"), focus.get("sub")] + list(focus.get("lines") or [])[:6]
        user = "You clicked this item in the 3D view: " + " | ".join(str(b) for b in bits if b) + "\n\n" + user
    rules = rule_texts(question, history)
    if rules:
        user = rules + "\n\n" + user
    if sources:  # the decisions this question is most likely about, right next to it (small models need this)
        user = ("MOST RELEVANT FACTS:\n" + "\n".join(f"- {s['item']}: built {s['built']}. Why: {s['why']}" for s in sources)
                + "\n\n" + user)
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user}]

    t0 = time.time()
    model = _pick_model()
    if model:
        try:
            text = _ask(model, messages).strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            if text:
                return {"answer": text, "model": model, "seconds": round(time.time() - t0, 1), "sources": sources}
        except Exception as exc:
            log.warning("design chat via %s failed: %s", model, exc)
    # no model: the saved reasons themselves
    if sources:
        text = "\n".join(f"{s['item']}: {s['built']}. {s['why']}" for s in sources)
    else:
        text = ("I couldn't match that to a decision in this design. Try asking about the sink, hob, tall units, "
                "wall cabinets, lofts, corners or finishes.")
    if rules:
        text = rules.replace("RULE TEXT:\n", "") + ("\n\n" + text if sources else "")
    return {"answer": text, "model": None, "seconds": round(time.time() - t0, 1), "sources": sources}
