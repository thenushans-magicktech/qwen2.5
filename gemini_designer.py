"""Google Gemini as an alternative AI designer (same answer schema and rules as Qwen).

By default (``GEMINI_USE_KNOWLEDGE`` off) Gemini designs from its OWN judgement: it gets the
plain room facts, the cabinet list (code, size, catalogue type), the customer's needs and the
answer format - but none of the MagickHome knowledge pack (playbook, worked examples, cabinet
notes, rule hints, schema hint texts). With ``GEMINI_USE_KNOWLEDGE=1`` it gets exactly Qwen's
v2 prompt. Either way it answers in the same enum-locked JSON schema, so its choices are
validated, converted and solved by the same code path. Differences from Qwen:

* it runs in Google's cloud (needs ``GEMINI_API_KEY``; the room facts, codes and brief are
  sent to Google), and is much faster than Qwen on a laptop CPU;
* it "thinks" before answering; with ``includeThoughts`` the API returns a summary of that
  thinking, kept in the report as ``ai.thoughts``.

The key is read from the environment only and never written to files or reports.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Optional

import requests
from pydantic import ValidationError

import config

log = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_KEEP = {"type", "properties", "required", "items", "enum", "description", "minItems", "maxItems",
         "minimum", "maximum", "nullable", "anyOf"}


def gemini_schema(schema: dict, descriptions: bool = True) -> dict:
    """Pydantic JSON schema -> the JSON-schema subset Gemini accepts: $refs inlined,
    ``const`` -> ``enum``, titles/defaults/string lengths dropped (lengths are enforced
    after the reply), property order kept so the reasons come before the decisions.
    ``descriptions=False`` also drops the field descriptions (they carry design hints)."""
    defs = schema.get("$defs", {})
    keep = _KEEP if descriptions else _KEEP - {"description"}

    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
        node = dict(node)
        if "const" in node:
            node["enum"] = [node.pop("const")]
            node.setdefault("type", "string")
        out = {}
        for k, v in node.items():
            if k == "properties":
                out[k] = {name: walk(sub) for name, sub in v.items()}
                out["propertyOrdering"] = list(v)
            elif k in ("items", "anyOf"):
                out[k] = walk(v)
            elif k in keep:
                out[k] = v
        return out

    return walk({k: v for k, v in schema.items() if k != "$defs"})


def _clip_strings(data: dict, schema: dict) -> dict:
    """Shorten over-long reason strings to the schema's maxLength (Gemini may write more)."""
    for name, prop in schema.get("properties", {}).items():
        limit = prop.get("maxLength")
        if limit and isinstance(data.get(name), str) and len(data[name]) > limit:
            cut = data[name][:limit]
            data[name] = cut[: cut.rfind(" ")] if " " in cut[limit // 2:] else cut
    return data


class ModelUnavailable(RuntimeError):
    """This Gemini model is busy, out of quota or unreachable right now - try the next model."""


def _daily_quota_used(text: str) -> bool:
    """429 because the free daily request quota is used up (waiting a few seconds will not help)."""
    return "PerDay" in text or "per day" in text.lower()


def _post(model: str, body: dict, key: str) -> dict:
    """POST generateContent with back-off for temporary problems: Google overload (429/500/503)
    and dropped connections (SSL EOF, connection reset). Raises :class:`ModelUnavailable` when
    this model cannot answer now (still busy, daily quota used up, timed out), so the caller can
    move on to the next model; other HTTP errors raise ``RuntimeError``."""
    url = f"{API_BASE}/models/{model}:generateContent"
    waits = (4, 10, 20)
    reason = "unavailable"
    for i in range(len(waits) + 1):
        try:
            r = requests.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json=body, timeout=config.GEMINI_TIMEOUT_S)
        except requests.exceptions.Timeout as exc:  # a slow reply: another model is faster than waiting again
            raise ModelUnavailable(f"{model} did not answer within {config.GEMINI_TIMEOUT_S:.0f} s") from exc
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
            reason = f"{model}: connection dropped ({exc.__class__.__name__})"
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and _daily_quota_used(r.text):
                raise ModelUnavailable(f"{model}: free daily quota used up (resets at midnight Pacific time)")
            if r.status_code not in (429, 500, 503):
                raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:400]}")
            reason = f"{model}: busy (HTTP {r.status_code})"
        if i < len(waits):
            log.warning("Gemini %s - retrying in %ss", reason, waits[i])
            time.sleep(waits[i])
    raise ModelUnavailable(reason)


def model_chain(model: Optional[str] = None) -> list[str]:
    """The chosen model first, then the backup models (each has its own quota), without repeats."""
    first = model or config.GEMINI_MODEL
    return list(dict.fromkeys([first] + [m for m in config.GEMINI_FALLBACK_MODELS if m]))


def design_with_gemini(geometry, groups, prefs, version: Optional[str] = None, model: Optional[str] = None,
                       knowledge: Optional[bool] = None):
    """Return ``(proposal, info)`` like ``ai_designer.design_with_ollama``.
    ``knowledge`` None = ``config.GEMINI_USE_KNOWLEDGE`` (default off: own judgement)."""
    from ai_designer import AIUnavailable, build_messages

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise AIUnavailable("GEMINI_API_KEY is not set")
    version = version or config.AI_PROMPT_VERSION
    model = model or config.GEMINI_MODEL
    knowledge = config.GEMINI_USE_KNOWLEDGE if knowledge is None else knowledge
    messages, Model, to_proposal = build_messages(geometry, groups, prefs, version, knowledge=knowledge)
    schema = Model.model_json_schema()
    contents = [{"role": "user", "parts": [{"text": messages[1]["content"]}]}]
    gen = {
        "responseMimeType": "application/json",
        "responseJsonSchema": gemini_schema(schema, descriptions=knowledge),
        "maxOutputTokens": 16384,
        "thinkingConfig": {"includeThoughts": True},
    }
    errors: list[str] = []
    models, skipped = model_chain(model), []
    for attempt in range(config.AI_MAX_RETRIES + 1):
        body = {"systemInstruction": {"parts": [{"text": messages[0]["content"]}]}, "contents": contents,
                "generationConfig": gen}
        t0 = time.time()
        data = None
        while models:
            try:
                data = _post(models[0], body, key)
                break
            except ModelUnavailable as exc:  # busy / quota / timeout: the next model has its own capacity
                skipped.append(str(exc))
                log.warning("Gemini %s - trying the next model", exc)
                models.pop(0)
            except Exception as exc:
                raise AIUnavailable(f"Gemini request failed: {exc}") from exc
        if data is None:
            raise AIUnavailable("Gemini unavailable - every model failed: " + "; ".join(skipped))
        cand = (data.get("candidates") or [{}])[0]
        parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        thoughts = "\n".join(p.get("text", "") for p in parts if p.get("thought")).strip()
        try:
            design = Model.model_validate(_clip_strings(json.loads(text), schema))
        except (ValueError, ValidationError) as exc:
            errors.append(f"{cand.get('finishReason')}: {str(exc)[:300]}")
            contents += [{"role": "model", "parts": [{"text": text or "{}"}]},
                         {"role": "user", "parts": [{"text": f"Invalid JSON for the schema: {str(exc)[:500]}. Reply again."}]}]
            continue
        usage = data.get("usageMetadata", {})
        info = {
            "mode": "gemini",
            "model": data.get("modelVersion") or models[0],
            "model_notes": skipped,  # models skipped before this one answered (busy, quota, ...)
            "prompt_version": version,
            "knowledge": knowledge,
            "attempts": attempt + 1,
            "seconds": round(time.time() - t0, 1),
            "prompt_tokens": usage.get("promptTokenCount"),
            "eval_count": usage.get("candidatesTokenCount"),
            "thinking_tokens": usage.get("thoughtsTokenCount"),
            "thoughts": thoughts,
            "raw": json.loads(design.model_dump_json()),
        }
        proposal = to_proposal(design)
        proposal.source = "gemini"
        return proposal, info
    raise AIUnavailable(f"Gemini reply failed schema validation: {errors[-1] if errors else 'unknown'}")
