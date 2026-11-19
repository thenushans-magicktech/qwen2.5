# Qwen designer benchmark

Model `qwen2.5-coder:3b`, AI mode `ollama`, run 2026-11-19 14:48.
Score = share of applicable designer checks that Qwen's own decisions pass (before the rule engine corrects them).

| Room | v2 score | v2 fixes | v2 time | v1 score | v1 fixes | v1 time |
|---|---|---|---|---|---|---|
| reference | 86% (12/14) | 0 | 182s | 100% (13/13) | 0 | 170s |
| compact | 92% (11/12) | 0 | 96s | 92% (11/12) | 1 | 142s |
| window_short_wall | 86% (12/14) | 0 | 104s | 85% (11/13) | 1 | 159s |
| door_by_window_corner | 79% (11/14) | 2 | 101s | 77% (10/13) | 0 | 178s |
| no_window | 100% (12/12) | 0 | 114s | 92% (11/12) | 0 | 156s |
| big_U | 93% (13/14) | 0 | 147s | 86% (12/14) | 1 | 151s |
| window_near_corner | 93% (13/14) | 1 | 122s | 71% (10/14) | 2 | 156s |
| narrow_galley | 67% (8/12) | 3 | 102s | 75% (9/12) | 0 | 163s |

- **v2**: average score 86.8%, 0.8 solver corrections per design, 121s per design, 7/8 scenes pass verification.
- **v1**: average score 84.6%, 0.6 solver corrections per design, 159s per design, 8/8 scenes pass verification.

## Failed checks per version

- **v2**: tall_well_placed x5, reads_room x2, hob_clear_of_door x2, hob_position_sane x2, features_zoned x2, wall_cabinets_planned x1
- **v1**: sink_size_suits_wall x5, walls_connected x4, features_zoned x3, tall_well_placed x1, wall_cabinets_planned x1, hob_position_sane x1, reads_room x1

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
- **v2** (86%, walls Wall 1, Wall 2): Window Wall: Wall 2, Door Wall: Wall 4, Door Corner: Wall 4, Long Solid Walls: Wall 1, Wall 3 WET zone: Sink (SINK-900) centred at 1991 mm on Wall 2. COOKING zone: Hob (HOB-900-3DRAWER) on Wall 1, centered at 2150 mm. Tall STORAGE zone: Tall pantry (PANTRY-600) at the end of Wall 1 near the entrance. WORK TRIANGLE: Sink (SINK-900) -> Hob (HOB-900-3DRAWER) -> Tall pantry (PANTRY-600).
  - failed: reads_room, tall_well_placed; purifier yes; talls: Tall Unit; corrections: none
- **v1** (100%, walls Wall 1, Wall 2, Wall 3): Wall 2 (id wall_56i56ymve0xf69fm) has the window, Wall 1 (id wall_1oy5ecjq6k11pfoo) has the door, Wall 3 (id wall_49fdugi4g7vj12pf) and Wall 4 (id wall_7sna3rldnsuwipgy) are solid and long. The work triangle is established with the sink on Wall 2, the hob on Wall 3, and the tall storage on Wall 4. The sink is centred under the window, and the hob is on a solid wall with a long free run, ensuring the chimney can be ducted and the cook is not in the door's path.
  - failed: none; purifier no; talls: Tall Unit; corrections: none

### compact
- **v2** (92%, walls Wall 1, Wall 2): Window Wall: Wall 2, Door Wall: Wall 4, Door Corner: Wall 1, Long Solid Walls: Wall 1, Wall 2 WET zone: Sink (SINK-900) at 1350 mm on Wall 2, Wet Zone: Purifier (AQUAGUARD-600) beside the window. COOKING zone: Hob (HOB-900-3DRAWER) at 1200 mm on Wall 1, Hob Center: 1200 mm, Hob Wall: Wall 1, Hob Centre Range: 1050-1350 mm. STORAGE zone: Tall Storage (TALL-OVEN-600) at the end of Wall 1, near the entrance. Work Triangle: Sink (SINK-900) -> Hob (HOB-900-3DRAWER) -> Tall Storage (TALL-OVEN-60
  - failed: tall_well_placed; purifier yes; talls: Tall Built-In Wicker Basket; corrections: none
- **v1** (92%, walls Wall 1, Wall 2, Wall 3): Wall 2 (id wall_56i56ymve0xf69fm) has the window, Wall 4 (id wall_7sna3rldnsuwipgy) has the door, Walls 1 and 3 are solid and long. The work triangle is sink (on Wall 2) - hob (on Wall 3) - tall storage (on Wall 1).
  - failed: sink_size_suits_wall; purifier no; talls: Tall Built-In Oven; corrections: tall unit(s) dropped: they left no legal hob position

### window_short_wall
- **v2** (86%, walls Wall 1, Wall 2): Window Wall: Wall 2, Door Wall: Wall 3, Door Corner: Near the Wall 3 corner, Long Solid Walls: Wall 1, Wall 3 WET zone: Sink (SINK-900) centred at 1500 on Wall 2. COOKING zone: Hob (HOB-900-3DRAWER) on Wall 1, centered at 1700. Tall STORAGE zone: Pantry (PANTRY-600) at the end of Wall 1 near the entrance. WORK TRIANGLE: Sink (SINK-900) -> Hob (HOB-900-3DRAWER) -> Pantry (PANTRY-600).
  - failed: reads_room, wall_cabinets_planned; purifier yes; talls: Tall Unit; corrections: none
- **v1** (85%, walls Wall 1, Wall 2, Wall 3): Wall 2 (id wall_56i56ymve0xf69fm) has the window, Wall 4 (id wall_7sna3rldnsuwipgy) has the door, Wall 1 (id wall_1oy5ecjq6k11pfoo) and Wall 3 (id wall_49fdugi4g7vj12pf) are solid and long. The work triangle is established with the sink on Wall 2, the hob on Wall 1, and the tall storage on Wall 3.
  - failed: sink_size_suits_wall, walls_connected; purifier no; talls: Tall Built-In Oven; corrections: added window wall Wall 2 so the sink can sit under the window

### door_by_window_corner
- **v2** (79%, walls Wall 1, Wall 2): Window Wall: Wall 2, Door Wall: Wall 1, Door Corner: Wall 2, Long Solid Walls: Wall 1, Wall 3 WET zone: Sink (SINK-900) centred at 1800 mm on Wall 2. COOKING zone: Hob (HOB-900-3DRAWER) on Wall 1, farther from the door. Tall STORAGE zone: Tall pantry (PANTRY-600) at the free end of Wall 1 near the entrance. WORK TRIANGLE: Sink (SINK-900) -> Hob (HOB-900-3DRAWER) -> Tall pantry (PANTRY-600).
  - failed: hob_clear_of_door, hob_position_sane, tall_well_placed; purifier yes; talls: Tall Unit; corrections: cylinder unit dropped: no room beside the hob with chimney/sink clearances; display unit skipped: no free run end for it
- **v1** (77%, walls Wall 2, Wall 3): Wall 2 (id wall_56i56ymve0xf69fm) has the window, Wall 1 (id wall_1oy5ecjq6k11pfoo) has the door, and Walls 3 and 4 (id wall_49fdugi4g7vj12pf and id wall_7sna3rldnsuwipgy) are solid and long. The work triangle is established with the sink on Wall 2, the hob on Wall 3, and the tall storage on Wall 4.
  - failed: sink_size_suits_wall, tall_well_placed, wall_cabinets_planned; purifier no; talls: Tall Built-In Oven; corrections: none

### no_window
- **v2** (100%, walls Wall 1, Wall 2, Wall 3): Window Wall: Wall 1, Door Wall: Wall 4, Door Corner: Wall 1, Long Solid Walls: Wall 1, Wall 2, Wall 3 WET zone: Sink (SINK-900) under the window, Purifier (AQUAGUARD-600) beside the window, Wicker Basket (WICKER-600) near the sink. COOKING zone: Hob (HOB-900-3DRAWER) on Wall 2, far from the door, Spices (SPICE-300) beside the hob. STORAGE zone: Tall Pantry (PANTRY-600) at the end of Wall 1, near the entrance. Work Triangle: Sink (SINK-900) -> Hob (HOB-900-3DRAWER) -> Tall Pantry (PANTRY-600).
  - failed: none; purifier yes; talls: Tall Unit; corrections: none
- **v1** (92%, walls Wall 1, Wall 2, Wall 3): Window: Wall 1, Door: Wall 4, Solid and Long Walls: Wall 1, Wall 2, Wall 3 Sink: Centred under the window on Wall 1. Hob: On Wall 2, far end of the long run. Tall Storage: On Wall 3, far end of the long run.
  - failed: sink_size_suits_wall; purifier no; talls: Tall Unit; corrections: none

### big_U
- **v2** (93%, walls Wall 1, Wall 2, Wall 3): Window Wall: Wall 2, Door Wall: Wall 4, Door Corner: Wall 1, Long Solid Walls: Wall 1, Wall 3 WET zone: Sink SINK-1050 on Wall 2, Wet Zone: Purifier AQUAGUARD-600 on Wall 2, Wet Zone: Wicker WICKER-600 near the sink. COOKING zone: Hob HOB-900-3DRAWER on Wall 1, Hob Center: 1700 mm, Hob Center Range: 1100-2500 mm. STORAGE zone: Tall Storage TALL-WICKER-600 on Wall 3, Tall Storage TALL-OVEN-600 on Wall 3. WORK TRIANGLE: Sink SINK-1050 on Wall 2, Hob HOB-900-3DRAWER on Wall 1, Tall Storage T3
  - failed: features_zoned; purifier yes; talls: Tall Built-In Wicker Basket; corrections: none
- **v1** (86%, walls Wall 1, Wall 2, Wall 3): Wall 2 (id wall_56i56ymve0xf69fm) has the window, Wall 4 (id wall_7sna3rldnsuwipgy) has the door, Wall 1 (id wall_1oy5ecjq6k11pfoo) and Wall 3 (id wall_49fdugi4g7vj12pf) are solid and long. The work triangle is established with the sink on Wall 2, the hob on Wall 1, and the tall storage on Wall 3.
  - failed: walls_connected, features_zoned; purifier no; talls: Tall Built-In Oven; corrections: added window wall Wall 2 so the sink can sit under the window

### window_near_corner
- **v2** (93%, walls Wall 1, Wall 2): Window Wall: Wall 2, Door Wall: Wall 4, Door Corner: Wall 1, Long Solid Walls: Wall 1, Wall 3 WET zone: Sink (SINK-900) at 2900 mm on Wall 2, Wet Zone: Purifier (AQUAGUARD-600) beside the window. COOKING zone: Hob (HOB-900-3DRAWER) at 1700 mm on Wall 1, Hob Center: 1700 mm, Hob Wall: Wall 1, Hob Centre Range: 1100-2200 mm. Tall Storage: Pantry (PANTRY-600) at the end of Wall 1, near the entrance.
  - failed: tall_well_placed; purifier yes; talls: Tall Unit; corrections: hob moved to Wall 2 (clearances on proposed wall failed)
- **v1** (71%, walls Wall 1, Wall 2): Wall 2 (id wall_56i56ymve0xf69fm) has the window, Wall 4 (id wall_7sna3rldnsuwipgy) has the door, Wall 1 (id wall_1oy5ecjq6k11pfoo) and Wall 3 (id wall_49fdugi4g7vj12pf) are solid and long. The work triangle is established with the sink on Wall 2, the hob on Wall 1, and the tall storage on Wall 3.
  - failed: hob_position_sane, sink_size_suits_wall, walls_connected, features_zoned; purifier no; talls: Tall Unit; corrections: added window wall Wall 2 so the sink can sit under the window; tall unit(s) dropped: they left no legal hob position

### narrow_galley
- **v2** (67%, walls Wall 2): Window Wall: Wall 2, Door Wall: Wall 1, Door Corner: Wall 2, Long Solid Walls: Wall 1, Wall 3 WET zone: Sink (SINK-900) at 2400 mm, Purifier (AQUAGUARD-600) beside the window. COOKING zone: Hob (HOB-900-3DRAWER) at 1200 mm, Spice Pull-out (SPICE-300) right beside the hob. STORAGE zone: Tall Storage (TALL-OVEN-600) at the end of Wall 1 near the entrance. Work Triangle: Sink (SINK-900) -> Hob (HOB-900-3DRAWER) -> Tall Storage (TALL-OVEN-600).
  - failed: hob_clear_of_door, hob_position_sane, tall_well_placed, features_zoned; purifier yes; talls: Tall Built-In Wicker Basket; corrections: hob moved to Wall 2 (clearances on proposed wall failed); tall unit(s) dropped: they left no legal hob position; display unit skipped: no free run end for it
- **v1** (75%, walls Wall 2, Wall 4): Wall 2 (id wall_56i56ymve0xf69fm) has the window, Wall 4 (id wall_7sna3rldnsuwipgy) has the door, and Walls 1 and 3 are solid and long. The sink goes on Wall 2, centred under the window. The hob goes on Wall 4, on a different wall from the sink, to allow for a long free run and a chimney duct. The tall storage goes at the far end of the run on Wall 4.
  - failed: reads_room, walls_connected, features_zoned; purifier no; talls: Tall Unit; corrections: none
