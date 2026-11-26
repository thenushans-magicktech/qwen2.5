"""Chat examples: teach the fine-tuned model to talk as the MagickHome kitchen designer assistant.

    python finetune/chat_dataset.py          # -> finetune/data/chat.jsonl + finetune/chat_review.md

Every answer comes from the MagickHome knowledge files, so the model learns MagickHome's own rules and words:
  * identity        - who it is, what it can and cannot do (and that real designs come from the designer app);
  * rule book       - every rule of knowledge/Kitchen_Rule_Engine_V1.xlsx ("What does rule R12 say?", and for the
                      module-width rules natural questions like "How wide is the hob in a Medium L-shape kitchen?");
  * playbook        - the "why" behind placements (knowledge/designer_playbook.md), written as Q&A;
  * cabinet guide   - when to choose / avoid each cabinet type (knowledge/cabinet_guide.json).

The examples have NO system message, so plain chat (e.g. the llama.cpp chat page) gets the MagickHome persona.
The playbook answers are drafts written from the playbook: MagickHome designers should review and edit
finetune/chat_review.md, and hand-written Q&A can be added to finetune/samples/chat_*.json (see README).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import common  # noqa: F401  (sets sys.path)
from common import DATA, ROOT
import rulebook

OUT = DATA / "chat.jsonl"
REVIEW = Path(__file__).resolve().parent / "chat_review.md"
EXTRA = Path(__file__).resolve().parent / "samples"   # chat_*.json: hand-written Q&A [{"q": ..., "a": ...}]

DESIGNER_HINT = ("To get a full design, upload your floor plan in the MagickHome Kitchen Designer - it lays out real "
                 "MagickHome cabinets and checks every rule.")

IDENTITY = [
    (["Who are you?", "who are you", "What are you?", "Introduce yourself."],
     "I'm the MagickHome kitchen designer assistant - an AI fine-tuned by MagickTech on MagickHome kitchen designs, "
     "the MagickHome Kitchen Rule Engine and the MagickHome cabinet catalogue. I'm built on an open-source language "
     "model. I can explain the design rules, help you choose cabinets and talk through layout ideas. " + DESIGNER_HINT),
    (["Are you fine-tuned on kitchen designs?", "Were you trained for kitchens?", "are you finetuned with kitchen designs?"],
     "Yes. I was fine-tuned on MagickHome kitchen designs that follow the MagickHome Kitchen Rule Engine - which walls "
     "carry cabinets, where the sink and hob go, which units a household needs - so I design and explain kitchens the "
     "MagickHome way. " + DESIGNER_HINT),
    (["What can you do?", "How can you help me?", "What do you help with?"],
     "I can: explain the MagickHome rules (sizes, clearances, where the sink, hob and tall units go); help you choose "
     "cabinets (sink, hob, drawers, tall units, purifier, lofts, wall doors); explain why a cabinet cannot go "
     "somewhere; and suggest layouts (L, U, single wall). " + DESIGNER_HINT),
    (["Can you see my floor plan?", "Can you design my kitchen here in chat?"],
     "Not in this chat - I can't see your room here. Upload your floor plan (MagickHomes scene file) in the MagickHome "
     "Kitchen Designer: it reads the walls, doors and windows, and I design the kitchen there with the full rule check. "
     "Here I can answer questions about the rules and cabinets."),
    (["Which company made you?", "Who built you?"],
     "I was fine-tuned by MagickTech for MagickHome kitchens, on top of an open-source language model."),
]

# Playbook Q&A (drafts from knowledge/designer_playbook.md - for designers to review)
PLAYBOOK = [
    (["Why is the sink placed under the window?", "Why should the sink go under the window?"],
     "In MagickHome kitchens the sink is centred under the window: you get daylight and fresh air while washing, and a "
     "view. That makes the window wall the wet zone; a wall water purifier, if needed, hangs on the same wall beside the window."),
    (["Why can't the hob go on a wall with a window?", "Can I put the hob under the window?"],
     "The hob and chimney go on a solid wall with no window: wind from a window can blow the flame out, and the "
     "chimney needs a solid wall to mount on and vent from."),
    (["Can I put the hob right next to the sink?", "Why not keep the hob beside the sink?"],
     "Better not. In L and U kitchens MagickHome puts the sink and the hob on different walls. If they must share a "
     "wall, keep at least 600-900 mm of counter between them - that is your preparation space, and it keeps water away "
     "from the flame."),
    (["How far should the hob be from a door?", "Can the hob be near the door?"],
     "Keep the hob at least 600 mm from a door, out of the corners, with at least 300 mm of counter on both sides. "
     "A door next to the hob is a safety risk and a draught on the flame."),
    (["Where do tall units go?", "Where should the tall pantry be placed?"],
     "Tall units (pantry, built-in oven or microwave) go at the END of a run, ideally the end nearest the entrance. "
     "Never in front of a window and never right beside the hob."),
    (["Can I put a tall unit next to the hob?", "Why not keep the pantry beside the hob?"],
     "No - a tall unit right beside the hob blocks the counter you need while cooking and gets heat and grease on its "
     "side. Put it at the end of a run, ideally near the entrance."),
    (["What is the work triangle?", "Explain the kitchen work triangle."],
     "The work triangle links the sink, the hob and the tall storage. Each leg should be about 1200-2700 mm, and no leg "
     "should cross the door path - short enough to save steps, long enough for counter space."),
    (["When should I choose an L-shaped kitchen?", "When is an L-shape best?"],
     "L-shape suits most rooms: the window wall plus one adjacent solid wall. Put the hob on the adjacent wall that is "
     "farther from the door."),
    (["When should I choose a U-shaped kitchen?", "When is a U-shape possible?"],
     "Choose U (the window wall plus both adjacent walls) only when the aisle between the two side runs is at least "
     "1200 mm and both side walls have a useful free run. It gives the most storage and counter."),
    (["When is a single-wall kitchen used?", "When should the kitchen be on one wall?"],
     "A single-wall kitchen is for narrow rooms, or when the other walls are blocked by doors."),
    (["How much free wall does a cabinet run need?", "Can a short wall carry cabinets?"],
     "A wall with less than 1200 mm of free run cannot carry cabinets."),
    (["What sink size should I choose?", "Which sink unit size is right?"],
     "900 mm is the standard sink unit. Choose 1050 mm when the window wall has 3000+ mm free, 1200 mm (large or double "
     "bowl) only with 3600+ mm free, and 800 mm for small kitchens with under 2400 mm free. The rule book can also set "
     "the sink width by kitchen size."),
    (["What hob unit size should I choose?", "900 or 750 hob unit?"],
     "900 mm normally, 750 mm when the hob wall has under 2400 mm free (the rule book sets the exact width by kitchen "
     "type and size band)."),
    (["Should I choose a drawer or shutter hob unit?", "Why a drawer hob unit?"],
     "Prefer a drawer hob unit (3 or 2 drawers): heavy pans and pressure cookers lie flat and are easy to lift. A "
     "shutter hob unit wastes space."),
    (["Do I need a spice pull-out?", "What size spice pull-out?"],
     "Yes - one spice pull-out right beside the hob for masalas and oils. 300 mm is standard; 150 mm only in very tight runs."),
    (["Do I need an LPG cylinder unit?", "Is the gas cylinder unit included?"],
     "LPG is assumed unless you tell us you have piped gas (PNG) or induction (rule R11), so the cylinder unit is "
     "included. It sits on the hob's left, with the spice pull-out on its right (seen from the room)."),
    (["I have piped gas. Do I need the cylinder unit?", "We use induction - is the LPG unit needed?"],
     "No. With piped gas (PNG) or induction the cylinder unit is dropped (rule R11); that space becomes normal storage."),
    (["Should I add lofts?", "When are lofts possible?"],
     "Most Indian kitchens have lofts above the wall cabinets for big vessels, festival items and bulk stock. They need "
     "a ceiling of at least 2658 mm. Without lofts the wall and tall units are made taller automatically, so storage "
     "still reaches the ceiling."),
    (["Hinged shutters or lift-up doors for wall cabinets?", "Should I choose lift-up wall doors?"],
     "Hinged shutters are the standard and cost least. Lift-up (flap-up) doors suit a modern look, or narrow kitchens "
     "where open doors would hit heads."),
    (["Should I add an open shelf?", "Is an open shelf unit a good idea?"],
     "At most one, at a run end away from the hob, for daily jars, tea and coffee - and only if you like the open look. "
     "Open shelves gather grease and dust."),
    (["Where should the crockery unit go?", "Do I need a crockery unit?"],
     "A crockery unit (glass tall unit) suits families who host guests or want a showcase. It goes at the run end "
     "nearest the dining area - never beside the hob or the sink."),
    (["Should I add a water purifier unit?", "Where does the Aquaguard go?"],
     "Most Indian homes use a wall water purifier. Its cabinet hangs on the sink wall beside the window. Skip it if you "
     "use an under-sink RO."),
    (["Drawers or shutters for base cabinets?", "Which base cabinet style should I choose?"],
     "Drawers for heavy daily cooking or 5+ people (pans and vessels stay reachable); mixed (a cutlery drawer on top, a "
     "shelf below) for most homes; shutters for light cooking or a tight budget."),
    (["Where does the vegetable wicker basket go?", "Where should the grain trolley be?"],
     "The vegetable wicker basket sits near the sink (wash, then cut). The grain trolley belongs in the storage zone, at "
     "a run end away from the hob heat."),
    (["What goes in the corners?", "Can I put a normal cabinet in the corner?"],
     "Corners take only corner cabinets (blind corner or L-corner units); a normal cabinet never goes in a corner. The "
     "designer adds them automatically."),
    (["How much counter is needed beside the hob and sink?", "What is the minimum counter at the end of a run?"],
     "Keep at least 450 mm of counter between the hob or the sink and the end of its run, and at least 300 mm on both "
     "sides of the hob."),
    (["What are the kitchen size bands?", "How is a kitchen classed as small, medium or large?"],
     "By floor area: Small 60-80 sqft, Medium 81-120 sqft, Large 121-180 sqft. The band decides module widths in the rule book."),
    (["How many hobs and sinks can a kitchen have?"],
     "One hob, one sink and one spice pull-out per kitchen."),
    (["Which wall gets the hob if I choose primary and secondary walls?"],
     "Your primary wall carries the hob and your secondary wall carries the sink."),
]

WIDTH_RE = re.compile(r"In an? (Small|Medium|Large) ([A-Za-z -]+?) kitchen \(([^)]*)\), set the (.+?) width to (\d+) mm", re.I)


def rule_qa(rb) -> list[tuple[list[str], str, str]]:
    out = []
    for n in sorted(rb.texts):
        text = (rb.texts[n] or "").strip()
        if not text:
            continue
        out.append(([f"What does rule R{n} say?", f"Explain rule R{n}."],
                    f"Rule R{n} of the MagickHome Kitchen Rule Engine: {text}", f"rule R{n}"))
        m = WIDTH_RE.search(text)
        if m:
            band, ktype, area, unit, width = m.groups()
            unit_l = unit.lower().replace("base ", "")
            out.append(([f"How wide is the {unit_l} in a {band.lower()} {ktype.strip()} kitchen?",
                         f"What {unit_l} width for a {band} {ktype.strip()} kitchen ({area})?"],
                        f"{width} mm. In a {band} {ktype.strip()} kitchen ({area}) MagickHome sets the {unit} width to "
                        f"{width} mm (rule R{n}).", f"rule R{n} width"))
    return out


def cabinet_qa() -> list[tuple[list[str], str, str]]:
    guide = json.loads((ROOT / "knowledge" / "cabinet_guide.json").read_text(encoding="utf-8"))
    cats = guide["categories"]
    items = cats.items() if isinstance(cats, dict) else [(c.get("category") or c.get("name"), c) for c in cats]
    out = []
    for name, c in items:
        if not name or c.get("status") not in (None, "offered"):
            continue
        parts = [c.get("purpose", "")]
        if c.get("choose_when"):
            parts.append("Choose it when: " + c["choose_when"])
        if c.get("avoid"):
            parts.append("Avoid: " + c["avoid"])
        ans = " ".join(p.strip() for p in parts if p and p.strip())
        if ans:
            out.append(([f"When should I use a {name}?", f"What is a {name} for?"], ans, f"cabinet {name}"))
    return out


def extra_qa() -> list[tuple[list[str], str, str]]:
    out = []
    for p in sorted(EXTRA.glob("chat_*.json")):
        for qa in json.loads(p.read_text(encoding="utf-8")):
            qs = qa["q"] if isinstance(qa["q"], list) else [qa["q"]]
            out.append((qs, qa["a"], f"designer {p.name}"))
    return out


def main() -> int:
    rb = rulebook.load()
    groups = [("identity", IDENTITY), ("playbook", PLAYBOOK)]
    blocks = [("identity", [(q, a, "identity") for q, a in IDENTITY]),
              ("playbook", [(q, a, "playbook") for q, a in PLAYBOOK]),
              ("rules", rule_qa(rb)), ("cabinets", cabinet_qa()), ("designer", extra_qa())]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    review = ["# Chat training examples - review", "",
              "Generated by `finetune/chat_dataset.py` from the knowledge files. Designers: correct any answer here by "
              "editing the source (playbook / workbook / cabinet guide / `chat_dataset.py`) or add hand-written Q&A in "
              "`finetune/samples/chat_*.json`, then re-run the script.", ""]
    with OUT.open("w", encoding="utf-8") as f:
        for kind, qas in blocks:
            review += [f"## {kind} ({len(qas)})", ""]
            for i, (questions, answer, tag) in enumerate(qas):
                for k, q in enumerate(questions):
                    row = {"id": f"chat-{kind}-{i:03d}-{k}", "source": "chat", "teacher": "knowledge", "score": None,
                           "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": answer}],
                           "meta": {"kind": kind, "tag": tag}}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
                review.append(f"- **{' / '.join(questions)}**  \n  {answer}")
            review.append("")
    REVIEW.write_text("\n".join(review), encoding="utf-8")
    counts = {k: sum(len(q) for q, _, _ in v) for k, v in blocks}
    print(f"{n} chat examples -> {OUT}  {counts}\nreview file: {REVIEW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
