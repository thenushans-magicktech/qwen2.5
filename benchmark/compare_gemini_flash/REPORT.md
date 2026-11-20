# Qwen designer benchmark

Model `gemini-flash-latest`, AI mode `ollama`, run 2026-11-20 10:01.
Score = share of applicable designer checks that Qwen's own decisions pass (before the rule engine corrects them).

| Room | gemini:gemini-flash-latest score | gemini:gemini-flash-latest fixes | gemini:gemini-flash-latest time |
|---|---|---|---|
| reference | 100% (14/14) | 0 | 118s |
| compact | 100% (12/12) | 1 | 168s |
| window_short_wall | 100% (14/14) | 0 | 36s |
| door_by_window_corner | 93% (13/14) | 0 | 40s |
| no_window | 100% (12/12) | 0 | 133s |

- **gemini:gemini-flash-latest**: average score 98.6%, 0.2 solver corrections per design, 99s per design, 5/5 scenes pass verification.

## Failed checks per version

- **gemini:gemini-flash-latest**: tall_well_placed x1

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
- **gemini:gemini-flash-latest** (100%, walls Wall 1, Wall 2, Wall 3): Wall 2 has a large window (1241-2741 mm), Wall 1 has an entrance door near the Wall 4 corner (85-985 mm), and Wall 3 and Wall 4 are long, uninterrupted solid walls. A U-shaped layout establishes distinct functional zones: tall and dry storage along Wall 1 by the entrance, the wet zone centred under the window on Wall 2, and the cooking zone along solid Wall 3,
  - failed: none; purifier yes; talls: Tall Unit; corrections: none

### compact
- **gemini:gemini-flash-latest** (100%, walls Wall 2, Wall 3): Wall 2 has a window (850-1850 mm, centre 1350 mm). Wall 4 has the entrance door (1600-2500 mm, near the Wall 1 corner). Wall 1 and Wall 3 are solid walls of 2400 mm length. An L-shaped layout with the wet zone and tall pantry on Wall 2 and the cooking zone on Wall 3. The work triangle flows smoothly from tall pantry to sink to hob without crossing entrance traffic.
  - failed: none; purifier yes; talls: Tall Unit; corrections: water purifier unit skipped: no free wall-cabinet space within reach of the sink (window, corner cabinets and chimney take the wall)

### window_short_wall
- **gemini:gemini-flash-latest** (100%, walls Wall 1, Wall 2, Wall 3): Wall 2 has a 1200 mm window centred at 1500 mm. Wall 4 has an entrance door near the Wall 3 corner (300-1200 mm). Wall 1 and Wall 3 are long solid walls of 4200 mm. A U-shaped layout zones cooking on Wall 1, wet prep and washing on Wall 2, and dry storage on Wall 3, creating an efficient work triangle with 1780 mm of clear central aisle space.
  - failed: none; purifier yes; talls: Tall Unit; corrections: none

### door_by_window_corner
- **gemini:gemini-flash-latest** (93%, walls Wall 2, Wall 3): Wall 2 has the window at 1200-2400 mm; Wall 1 has the door near the Wall 2 corner (2200-3100 mm); Wall 3 and Wall 4 are long solid walls. The wet zone centres on Wall 2 under the window, cooking is placed on adjacent solid Wall 3, and tall storage anchors the end of Wall 3, creating an efficient, uncrossed work triangle.
  - failed: tall_well_placed; purifier yes; talls: Tall Unit; corrections: none

### no_window
- **gemini:gemini-flash-latest** (100%, walls Wall 1, Wall 2): Wall 4 is the door wall with the door located near the Wall 1 corner; Walls 1, 2, and 3 are solid walls with no windows, with Wall 2 offering the longest run at 3600 mm. An L-shaped layout across Wall 1 and Wall 2 establishes an efficient work triangle, locating tall storage and the sink along Wall 1 and the cooking zone along Wall 2 safely away from the doorway.
  - failed: none; purifier yes; talls: Tall Unit; corrections: none
