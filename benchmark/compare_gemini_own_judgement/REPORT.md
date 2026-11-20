# Qwen designer benchmark

Model `gemini-flash-latest`, AI mode `ollama`, run 2026-11-20 11:06.
Score = share of applicable designer checks that Qwen's own decisions pass (before the rule engine corrects them).

| Room | gemini:gemini-flash-latest score | gemini:gemini-flash-latest fixes | gemini:gemini-flash-latest time |
|---|---|---|---|
| reference | 93% (13/14) | 0 | 218s |
| compact | error | - | - |
| window_short_wall | error | - | - |
| door_by_window_corner | error | - | - |
| no_window | error | - | - |

- **gemini:gemini-flash-latest**: average score 92.9%, 0.0 solver corrections per design, 218s per design, 1/1 scenes pass verification.

## Failed checks per version

- **gemini:gemini-flash-latest**: reads_room x1

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
- **gemini:gemini-flash-latest** (93%, walls Wall 2, Wall 3): A very spacious 4100x4100 mm square kitchen with a primary entrance on Wall 1 near the Wall 4 corner and a wide 1500 mm window centrally positioned on Wall 2. Walls 3 and 4 offer continuous unbroken runs. An elegant, highly functional L-shaped layout along Wall 2 and Wall 3 maximizes ergonomics and natural light, positioning the wet zone under the window and the cooking and tall appliance zone along
  - failed: reads_room; purifier yes; talls: Tall Built-In Oven & Micro, Tall Unit; corrections: none

### compact
- **gemini:gemini-flash-latest**: AIUnavailable: Gemini request failed: HTTPSConnectionPool(host='generativelanguage.googleapis.com', port=443): Read timed out. (read timeout=240.0)

### window_short_wall
- **gemini:gemini-flash-latest**: AIUnavailable: Gemini request failed: Gemini HTTP 429: {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, mode

### door_by_window_corner
- **gemini:gemini-flash-latest**: AIUnavailable: Gemini request failed: Gemini HTTP 429: {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, mode

### no_window
- **gemini:gemini-flash-latest**: AIUnavailable: Gemini request failed: Gemini HTTP 429: {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, mode
