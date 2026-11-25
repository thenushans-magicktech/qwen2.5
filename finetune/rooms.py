"""Random kitchen rooms and customer briefs for the dataset builder.

Rooms are rectangles built from the production sample scene (the same builder as
designer_benchmark.Room): 2.0-5.5 m walls, 0-2 windows, 1-2 doors, 2.4-3.0 m ceilings.
Briefs vary the customer needs the prompt shows (fuel, purifier, appliances, storage, layout ...),
so the model learns both "stated need -> obey" and "not stated -> decide and say why".
"""
from __future__ import annotations

import random
from typing import Optional

from common import KitchenPreferences  # noqa: F401  (sets sys.path)
from designer_benchmark import Room


def _place(rng: random.Random, wall_len: int, width: int, taken: list[tuple[int, int]], gap: int = 200,
           end_margin: int = 150) -> Optional[int]:
    """A centre (mm) for an opening of ``width`` on a wall, clear of the ``taken`` intervals."""
    for _ in range(40):
        lo, hi = end_margin + width // 2, wall_len - end_margin - width // 2
        if hi <= lo:
            return None
        c = rng.randrange(lo, hi + 1, 50) if hi - lo >= 50 else lo
        a, b = c - width // 2, c + width // 2
        if all(b + gap <= s or a - gap >= e for s, e in taken):
            taken.append((a, b))
            return c
    return None


def random_room(rng: random.Random, key: str) -> Room:
    width = rng.randrange(2000, 5501, 100)   # Wall 2 / Wall 4
    depth = rng.randrange(1800, 5001, 100)   # Wall 1 / Wall 3
    lengths = {1: depth, 2: width, 3: depth, 4: width}
    height = rng.choice([2.4, 2.5, 2.5, 2.6, 2.7, 2.7, 2.8, 3.0])
    taken: dict[int, list] = {w: [] for w in lengths}
    windows, doors = [], []
    for _ in range(rng.choices([0, 1, 2], weights=[10, 72, 18])[0]):
        wall = rng.randint(1, 4)
        w = rng.choice([600, 900, 1000, 1200, 1200, 1500, 1800])
        if w <= lengths[wall] - 600 and (c := _place(rng, lengths[wall], w, taken[wall])) is not None:
            windows.append((wall, c, w))
    for _ in range(rng.choices([1, 2], weights=[82, 18])[0]):
        wall = rng.randint(1, 4)
        w = rng.choice([750, 800, 900, 900, 1000])
        near_corner = rng.random() < 0.4  # doors often sit near a corner
        margin = 100 if near_corner else 150
        if w <= lengths[wall] - 300 and (c := _place(rng, lengths[wall], w, taken[wall], end_margin=margin)) is not None:
            doors.append((wall, c, w))
    title = f"{width / 1000:.1f} x {depth / 1000:.1f} m, {len(windows)} window(s), {len(doors)} door(s), {height} m high"
    return Room(key, title, width, depth, windows=windows, doors=doors, height_m=height)


def random_prefs(rng: random.Random, **fixed) -> KitchenPreferences:
    """A customer brief: mostly 'not stated' (the designer decides), with some stated needs."""
    pick = lambda opts, weights: rng.choices(opts, weights=weights)[0]  # noqa: E731
    p = dict(
        door_style=pick(["Slab (Plain)", "Shaker -1"], [80, 20]),
        layout=pick(["auto", "single", "L", "U"], [85, 5, 5, 5]),
        include_tall=rng.random() < 0.85,
        fuel=pick(["auto", "lpg", "png", "induction"], [60, 20, 12, 8]),
        water_purifier=pick(["auto", "yes", "no"], [65, 25, 10]),
        built_in=pick(["auto", "none", "microwave", "oven", "oven_microwave"], [60, 10, 15, 10, 5]),
        base_storage=pick(["auto", "drawers", "mixed", "shutters"], [70, 15, 10, 5]),
        cooking=pick(["auto", "heavy", "light"], [60, 30, 10]),
        household_size=pick([None, 2, 3, 4, 5, 6], [50, 10, 15, 12, 8, 5]),
        include_loft=pick([None, True, False], [75, 15, 10]),
        budget=pick(["auto", "essential", "premium", "luxury"], [70, 12, 12, 6]),
        display_unit=pick(["auto", "yes", "no"], [80, 10, 10]),
        crockery_unit=pick(["auto", "yes", "no"], [85, 10, 5]),
    )
    p.update(fixed)
    return KitchenPreferences(**p)
