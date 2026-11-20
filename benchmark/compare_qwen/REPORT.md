# Qwen designer benchmark

Model `qwen2.5-coder:7b`, AI mode `ollama`, run 2026-11-20 10:01.
Score = share of applicable designer checks that Qwen's own decisions pass (before the rule engine corrects them).

| Room | v2 score | v2 fixes | v2 time |
|---|---|---|---|
| reference | 100% (14/14) | 0 | 31s |
| compact | 92% (11/12) | 2 | 26s |
| window_short_wall | 93% (13/14) | 0 | 30s |
| door_by_window_corner | 93% (13/14) | 0 | 33s |
| no_window | 100% (12/12) | 0 | 30s |
| big_U | 100% (14/14) | 0 | 32s |
| window_near_corner | 100% (14/14) | 0 | 32s |
| narrow_galley | 92% (11/12) | 1 | 29s |

- **v2**: average score 96.2%, 0.4 solver corrections per design, 30s per design, 8/8 scenes pass verification.

## Failed checks per version

- **v2**: sink_size_suits_wall x2, tall_well_placed x2

## Checks

- `reads_room`: room reading names the window wall and the door wall correctly
- `sink_on_window_wall`: sink on the window wall
- `hob_on_solid_wall`: hob on a solid wall (when a long solid wall is available)
- `sink_hob_apart`: sink and hob on different walls, or 600+ mm counter between them
- `hob_clear_of_door`: hob 600+ mm from any door on its wall
- `hob_position_sane`: requested hob centre leaves 300+ mm counter both sides inside a free run
- `sink_size_suits_wall`: sink width suits the window wall (1050 needs 3000+, 1200 needs 3600+ mm free)
- `hob_unit_has_drawers`: drawer hob unit (pans lie flat)
- `spice_beside_hob`: spice pull-out chosen
- `tall_well_placed`: tall unit chosen on a cabinet wall, not crowding the hob wall
- `walls_connected`: cabinet walls form one connected run including the window wall
- `uses_the_room`: L or U when the window wall has a usable adjacent wall
- `features_zoned`: drawers on the hob wall, wicker by the sink, grain away from the hob
- `wall_cabinets_planned`: wall-cabinet sizes chosen, only for cabinet walls

## Qwen's reasoning per room

### reference
- **v2** (100%, walls Wall 1, Wall 2, Wall 3): Window wall: Wall 2, Door wall: Wall 1, Door near Wall 4 corner, Long solid walls: Wall 2, Wall 3, Wall 4 WET zone: Sink centred at 1991 on Wall 2. COOKING zone: Hob on Wall 1, Chimney on Wall 3. STORAGE zone: Tall pantry at the free end of Wall 3 near the entrance. Work triangle: Sink -> Hob -> Tall Pan.
  - failed: none; purifier no; talls: Tall Unit; corrections: none

### compact
- **v2** (92%, walls Wall 1, Wall 2, Wall 3): Window wall: Wall 2, Door wall: Wall 4, Door near: Wall 4 corner, Long solid walls: Wall 1 and Wall 3 WET zone: sink centred at 1350 on Wall 2. COOKING zone: hob on Wall 1, 1050 mm from the door. STORAGE zone: tall pantry at the free end of Wall 3 near the entrance.
  - failed: sink_size_suits_wall; purifier no; talls: Tall Unit; corrections: hob moved to Wall 3 (clearances on proposed wall failed); tall unit(s) dropped: they left no legal hob position

### window_short_wall
- **v2** (93%, walls Wall 1, Wall 2, Wall 3): Window wall: Wall 2, Door wall: Wall 4, Door near: Wall 3 corner, Long solid walls: Wall 1 and Wall 3 WET zone: sink centred at 1500 on Wall 2. COOKING zone: hob on Wall 1, far from the door. STORAGE zone: tall pantry at the free end of Wall 1 near the entrance.
  - failed: sink_size_suits_wall; purifier yes; talls: Tall Unit; corrections: none

### door_by_window_corner
- **v2** (93%, walls Wall 1, Wall 2, Wall 3): Window wall: Wall 2, Door wall: Wall 1, Door near Wall 1 corner, Long solid walls: Wall 3 and Wall 4 WET zone: Sink on Wall 2, COOKING zone: Hob on Wall 3, STORAGE zone: Tall pantry on Wall 4. Work triangle: Sink -> Hob -> Tall pantry.
  - failed: tall_well_placed; purifier no; talls: Tall Unit; corrections: none

### no_window
- **v2** (100%, walls Wall 1, Wall 2, Wall 3): Window wall: Wall 2, Door wall: Wall 4, Door near: Wall 4 corner, Long solid walls: Wall 1 and Wall 3 WET zone: Sink on Wall 2, COOKING zone: Hob on Wall 1, STORAGE zone: Tall pantry on Wall 3. Work triangle: Sink -> Hob -> Tall pantry.
  - failed: none; purifier no; talls: Tall Unit; corrections: none

### big_U
- **v2** (100%, walls Wall 1, Wall 2, Wall 3): Window wall: Wall 2, Door wall: Wall 4, Door near: Wall 4 corner, Long solid walls: Wall 1, Wall 3 WET zone: sink centred at 2250 on Wall 2. COOKING zone: hob on Wall 1, chimney above, spice pull-out beside hob. STORAGE zone: tall pantry at the free end of Wall 3 near the entrance.
  - failed: none; purifier no; talls: Tall Unit; corrections: none

### window_near_corner
- **v2** (100%, walls Wall 1, Wall 2, Wall 4): Window wall: Wall 2, Door wall: Wall 4, Door near: Wall 4 corner, Long solid walls: Wall 1 and Wall 3 WET zone: Sink on Wall 2, COOKING zone: Hob on Wall 1, STORAGE zone: Tall pantry on Wall 3. Work triangle: Sink -> Hob -> Tall pantry.
  - failed: none; purifier no; talls: Tall Built-In Oven; corrections: none

### narrow_galley
- **v2** (92%, walls Wall 2, Wall 3, Wall 4): Window wall: Wall 2, Door wall: Wall 1, Door near: Wall 2 corner, Long solid walls: Wall 2, Wall 3, Wall 4 WET zone: sink centred at 2400 on Wall 2. COOKING zone: hob on Wall 3, chimney above. STORAGE zone: tall pantry at the free end of Wall 3 near the entrance. Work triangle: sink -> hob -> tall storage.
  - failed: tall_well_placed; purifier no; talls: Tall Unit; corrections: hob moved to Wall 4 (clearances on proposed wall failed)
