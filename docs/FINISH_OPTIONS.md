# Finish options by cabinet and budget

Generated from the MagickHome textures API (api: 174 cabinet finishes, 16 worktops, 22 backsplashes) and `knowledge/finish_rules.json`.

> The budget-to-finish rules are the generator's **proposed** rules (approved for use). The MagickHome rule workbook defines no colour or finish rules; edit `knowledge/finish_rules.json` to change them.

## 1. Which cabinets take which finish

A design has two cabinet finishes plus the worktop and backsplash. Every cabinet takes the finish of its run; fillers, slabs, end panels and cover panels match the run they close.

| Finish choice | Cabinet categories that take it |
|---|---|
| **Base run** (base + tall) | Base Unit, Base Sink Unit, Base Hob Unit, Base Spice Pull Out, Base Cylinder Unit, Base Grain Trolley, Base Wicker Basket Unit, Blind corner unit L / R, Tall Unit, Tall Built-In Oven / Microwave / Oven & Micro / Wicker Basket, Tall Glass Unit (frame); base and tall fillers, slabs and side panels |
| **Wall run** (wall + loft) | Wall Unit, Wall Lift Up Unit, Wall L Corner, Wall Aquaguard Unit, Wall Glass / Shaker Glass Unit (frame), Wall Open Unit, Loft - Frame with Shutter, Loft L Corner; wall and loft fillers, slabs and panels |
| **Worktop** | the countertop over every base run |
| **Backsplash** | the wall between the worktop and the wall units |

Not chosen by the generator yet (the MagickHome default is kept): glass panels of glass units (38 glass finishes), handles (19 metal finishes), carcass, floor, appliance finishes.

## 2. What each budget allows

| Budget | Base run | Wall run | PU / acrylic runs | Finishes allowed | Default if the designer decides |
|---|---|---|---|---|---|
| Essential | Laminate | Laminate | max 0 | 136 | base Pure Legno Oak, wall Dove grey |
| Premium | Laminate / PU / Acrylic | Laminate / PU / Acrylic | max 1 | 174 | base Pure Legno Oak, wall Stromma White |
| Luxury | PU / Acrylic | PU / Acrylic | max 2 | 38 | base Cadet Blue, wall Stromma White |
| Not stated | any | any | any | 174 | base Cadet Blue, wall Stromma White |

- A finish you pick yourself is always used; if it breaks the tier it is kept and flagged.
- If the AI designer picks PU / acrylic on both runs in Premium, one run is changed back to laminate.
- Worktops and backsplashes are allowed in every budget (no budget rule for them yet).

## 3. Cabinet finishes

### Laminate - Patterns (6) - allowed in: Essential, Premium

| Code | Name | Finish | Brand | Colour |
|---|---|---|---|---|
| MHS178 | Bermuda Triangles | Textured | Merino | texture |
| MHS179 | Cafe Kindred | Textured | Merino | texture |
| MHS176 | Crochet Denim Splash | Textured | Merino | texture |
| MHS175 | Steel | Textured | Merino | texture |
| MHS180 | Tan Cambric | Textured | Merino | texture |
| MHS177 | Tramac Twill | Textured | Merino | texture |

### Laminate - Solids (73) - allowed in: Essential, Premium

| Code | Name | Finish | Brand | Colour |
|---|---|---|---|---|
| MHS082 | Adriatic Blue | Suede | Merino | #304246 |
| MHS109 | Autumn Leaf | High Gloss | Merino | #3D3C37 |
| MHS078 | Autumn Leaf | Suede | Merino | #514D42 |
| MHS095 | Berry Bunch | High Gloss | Merino | #FFF9ED |
| MHS085 | Bianco | Anti-fingerprint | Rehau | #FFFFFF |
| MHS118 | Black | Textured | Merino | #161616 |
| MHS103 | Black current | High Gloss | Merino | #32202E |
| MHS083 | Black Current | Suede | Merino | #392C33 |
| MHS055 | Blush | Suede | Merino | #B08676 |
| MHS090 | Boston Brick | Anti-fingerprint | Rehau | #745F5C |
| MHS069 | Brown | Suede | Merino | #3E3B36 |
| MHS088 | Brown Teepee | Anti-fingerprint | Rehau | #9E8A72 |
| MHS094 | Cappuccino | High Gloss | Merino | #FAFBED |
| MHS077 | Cappuccino | Suede | Merino | #80766C |
| MHS054 | Caramel | Suede | Merino | #522603 |
| MHS067 | Cardinal | Suede | Merino | #B1262D |
| MHS098 | Carnival | High Gloss | Merino | #481512 |
| MHS084 | Chalk Board | Suede | Merino | #4E6354 |
| MHS096 | Champagne | High Gloss | Merino | #321322 |
| MHS058 | Champagne | Suede | Merino | #F5EDBF |
| MHS117 | Chocolate | Textured | Merino | #261D16 |
| MHS108 | Cobble Stone | High Gloss | Merino | #242520 |
| MHS102 | Dark Citrus | High Gloss | Merino | #5F562B |
| MHS089 | Dark grey | Anti-fingerprint | Rehau | #8E7F7A |
| MHS087 | Desert Shadow | Anti-fingerprint | Rehau | #8D807A |
| MHS052 | Dove grey | Suede | Merino | #80837C |
| MHS114 | Fiesta Rose | High Gloss | Merino | #4E262E |
| MHS116 | Frosty White | Textured | Merino | #FFF9ED |
| MHS119 | Frosty White | Textured | Merino | #8F928B |
| MHS097 | Frosty White | High Gloss | Merino | #F5EDBF |
| MHS092 | Hilltop | Anti-fingerprint | Rehau | #73AC8B |
| MHS091 | Honey Gold | Anti-fingerprint | Rehau | #E0A201 |
| MHS106 | Irish Cream | High Gloss | Merino | #9B978C |
| MHS060 | Irish Cream | Suede | Merino | #9B978C |
| MHS075 | Khaki | Suede | Merino | #867A60 |
| MHS080 | Lavender | Suede | Merino | #C3B5C4 |
| MHS056 | Light Beige | Suede | Merino | #E6DBBF |
| MHS086 | Light Grey | Anti-fingerprint | Rehau | #EFE8D6 |
| MHS070 | Lilac | Suede | Merino | #4A556B |
| MHS081 | Loto Green | Suede | Merino | #96AC7B |
| MHS068 | Marigold | Suede | Merino | #FADB4D |
| MHS113 | Marker Board | High Gloss | Merino | #E1E3E2 |
| MHS104 | Midnight Blue | High Gloss | Merino | #141C27 |
| MHS063 | Misty Grey | Suede | Merino | #383838 |
| MHS093 | Moro | Anti-fingerprint | Rehau | #08090B |
| MHS066 | Parakeet | Suede | Merino | #C2993D |
| MHS073 | Peanut | Suede | Merino | #7B7C6E |
| MHS122 | Pearl Black | Textured | Merino | #393F3F |
| MHS110 | Pearl Black | High Gloss | Merino | #596261 |
| MHS072 | Pearl Black | Suede | Merino | #596261 |
| MHS062 | Pink Salt | Suede | Merino | #DFBCB8 |
| MHS064 | Prairie Green | Suede | Merino | #6F855F |
| MHS059 | Pumic Grey | Suede | Merino | #9C9D95 |
| MHS057 | Royal White | Suede | Merino | #E5E6D8 |
| MHS074 | Sandal | Suede | Merino | #887A5D |
| MHS105 | Shangrilla | High Gloss | Merino | #310A0B |
| MHS100 | Shore | High Gloss | Merino | #1E3640 |
| MHS112 | Silver Grey | High Gloss | Merino | #837568 |
| MHS115 | Slate | Textured | Merino | #5D5D5D |
| MHS121 | Slate | Textured | Merino | #343530 |
| MHS124 | Slate | Textured | Merino | #353631 |
| MHS101 | Slate | High Gloss | Merino | #5D5D5D |
| MHS053 | Slate | Suede | Merino | #5D5D5D |
| MHS071 | Smoke Grey | Suede | Merino | #CED2D5 |
| MHS123 | Tan | Textured | Merino | #605448 |
| MHS099 | Tan | High Gloss | Merino | #504239 |
| MHS076 | Tan | Suede | Merino | #605448 |
| MHS065 | Toned Blue | Suede | Merino | #ABC1CE |
| MHS107 | Trooper | High Gloss | Merino | #112836 |
| MHS061 | Trooper | Suede | Merino | #112836 |
| MHS111 | White Lily | High Gloss | Merino | #E9F3F5 |
| MHS120 | Winter Sea | Textured | Merino | #444E50 |
| MHS079 | Winter Sea | Suede | Merino | #444E50 |

### Laminate - Stones (7) - allowed in: Essential, Premium

| Code | Name | Finish | Brand | Colour |
|---|---|---|---|---|
| MHS187 | Aggies Micas | Textured | Merino | texture |
| MHS186 | Allies Micas | Textured | Merino | texture |
| MHS181 | Ashen Krakato | Textured | Merino | texture |
| MHS183 | Barmer Dark Marble | Textured | Merino | texture |
| MHS184 | Golden Campino | Textured | Merino | texture |
| MHS182 | Pristine Wood | Textured | Merino | texture |
| MHS185 | Riviea Marble | Textured | Merino | texture |

### Laminate - Wooden (50) - allowed in: Essential, Premium

| Code | Name | Finish | Brand | Colour |
|---|---|---|---|---|
| MHS139 | American Vertical Walnut | Suede | Merino | texture |
| MHS145 | Apex Leaved Elm | Suede | Merino | texture |
| MHS165 | Arizona Olmo Walnut | Textured | Merino | texture |
| MHS173 | Austra Mondo Noce | Textured | Merino | texture |
| MHS171 | Authur Baltic Elm | Textured | Merino | texture |
| MHS133 | Balinese Fresco Pine | Suede | Merino | texture |
| MHS164 | Brazilian Vertical Walnut | Textured | Merino | texture |
| MHS127 | Canadian Walnut | Suede | Merino | texture |
| MHS163 | Chamois Tropical | Textured | Merino | texture |
| MHS158 | Cherry Cadiz Walnut | High Gloss | Merino | texture |
| MHS136 | Clammy Samsara Maple | Suede | Merino | texture |
| MHS148 | Classic Ashwood | Suede | Merino | texture |
| MHS166 | Cleave Baltic Elm | Textured | Merino | texture |
| MHS141 | Daco Stad Oak | Suede | Merino | texture |
| MHS174 | Deep Herring Wood | Textured | Merino | texture |
| MHS167 | Derwent Moca Walnut | Textured | Merino | texture |
| MHS159 | Doredos Cottage Pine | High Gloss | Merino | texture |
| MHS131 | Douglas Fresco Pine | Suede | Merino | texture |
| MHS160 | Ebony Sepia | High Gloss | Merino | texture |
| MHS157 | Elevated Wood | High Gloss | Merino | texture |
| MHS146 | Fondle Samsara Maple | Suede | Merino | texture |
| MHS156 | Garden Walnut | Anti-fingerprint | Rehau | texture |
| MHS137 | Gigan Lowa Walnut | Suede | Merino | texture |
| MHS140 | Havel Pearwood | Suede | Merino | texture |
| MHS155 | Hickory | Anti-fingerprint | Rehau | texture |
| MHS129 | Laurel Oak Whirl | Suede | Merino | texture |
| MHS168 | Magas Irish Walnut | Textured | Merino | texture |
| MHS143 | Magas Mystique Walnut | Suede | Merino | texture |
| MHS172 | Magas Walnut Bronze | Textured | Merino | texture |
| MHS142 | Magas Walnut Bronze | Suede | Merino | texture |
| MHS149 | Natural Ashwood | Suede | Merino | texture |
| MHS151 | Nice Pear | Suede | Merino | texture |
| MHS134 | Nice Wood | Suede | Merino | texture |
| MHS138 | Nogal Olmo Walnut | Suede | Merino | texture |
| MHS162 | Novoceno Pine | High Gloss | Merino | texture |
| MHS144 | Nutmeg Bristling Walnut | Suede | Merino | texture |
| MHS150 | Oder Pearwood | Suede | Merino | texture |
| MHS130 | Pabela Acacia Exodus | Suede | Merino | texture |
| MHS135 | Pure Legno Oak | Suede | Merino | texture |
| MHS154 | Red Oxford Cherry | Suede | Merino | texture |
| MHS161 | Sapelle | High Gloss | Merino | texture |
| MHS128 | Sapelle | Suede | Merino | texture |
| MHS132 | Savannah Cherry | Suede | Merino | texture |
| MHS125 | Siberian Pike Oak | Suede | Merino | texture |
| MHS152 | Silver Legno Oak | Suede | Merino | texture |
| MHS153 | Slam Wood | Suede | Merino | texture |
| MHS126 | Straight Walnut | Suede | Merino | texture |
| MHS170 | Tahiti Samoa Teak | Textured | Merino | texture |
| MHS169 | Toasted Herringbone | Textured | Merino | texture |
| MHS147 | Wych Koawood | Suede | Merino | texture |

### PU - Solids (20) - allowed in: Premium, Luxury

| Code | Name | Finish | Brand | Colour |
|---|---|---|---|---|
| MHS216 | Augusta Pink | Matte | Magickhome | #AE6C78 |
| MHS207 | Black | High Gloss | Magickhome | #353537 |
| MHS215 | Blue Haze | Matte | Magickhome | #030035 |
| MHS224 | Cadet Blue | Matte | Magickhome | #3B5A6E |
| MHS223 | Cinnamon Roe | Matte | Magickhome | #BA6747 |
| MHS213 | Erdbeerrot | Matte | Magickhome | #C73F4B |
| MHS221 | Faded Fern | Matte | Magickhome | #7E9978 |
| MHS214 | Island Green | Matte | Magickhome | #577048 |
| MHS225 | Jade Green | Matte | Magickhome | #007771 |
| MHS217 | Jasper Red | Matte | Magickhome | #A84D4C |
| MHS208 | Light Grey | Matte | Magickhome | #C0C1BC |
| MHS210 | Periwinkle Blue | Matte | Magickhome | #90A1BB |
| MHS220 | Quicksand | Matte | Magickhome | #927A62 |
| MHS211 | Red Wine | Matte | Magickhome | #31070B |
| MHS219 | Sea Foam | Matte | Magickhome | #8ED5CF |
| MHS218 | Spring Ave | Matte | Magickhome | #C9CE94 |
| MHS209 | Stratton Grey | Matte | Magickhome | #838486 |
| MHS212 | Stromma White | Matte | Magickhome | #F2F2F2 |
| MHS222 | Terranova | Matte | Magickhome | #6C4E46 |
| MHS206 | White | High Gloss | Magickhome | #FFFFFF |

### Acrylic - Solids (18) - allowed in: Premium, Luxury

| Code | Name | Finish | Brand | Colour |
|---|---|---|---|---|
| MHS190 | Azucar | High Gloss | Rehau | texture |
| MHS199 | Biscotti | High Gloss | Rehau | #BCAD9A |
| MHS196 | Black Rice | High Gloss | Rehau | texture |
| MHS205 | Blackberry | High Gloss | Rehau | #231F20 |
| MHS193 | Blue Spirulina | High Gloss | Rehau | #7A9BA4 |
| MHS197 | Creme | High Gloss | Rehau | #FCFAE5 |
| MHS204 | Dark Truffle | High Gloss | Rehau | #606165 |
| MHS203 | Earl Grey | High Gloss | Rehau | #848388 |
| MHS189 | French Vanilla | High Gloss | Rehau | #ECE9E2 |
| MHS194 | Gold Dust | High Gloss | Rehau | texture |
| MHS192 | Grey Flesh | High Gloss | Rehau | #AAB3B0 |
| MHS188 | Marshmallow | High Gloss | Rehau | #F5F4F0 |
| MHS202 | Pink Oyster | High Gloss | Rehau | #867889 |
| MHS201 | Roasted Chestnut | High Gloss | Rehau | #807D76 |
| MHS191 | Seafoam | High Gloss | Rehau | #C0D3C0 |
| MHS200 | Shiitake | High Gloss | Rehau | #B1AB9D |
| MHS195 | Silver Dust | High Gloss | Rehau | texture |
| MHS198 | Sweet Latte | High Gloss | Rehau | #FEF7C9 |

## 4. Worktops (16)

| Code | Name | Material | Brand |
|---|---|---|---|
| MHC001 | Armani Grey Quartz | Granite | Nil |
| MHC003 | Coffee Brown | Granite | Nil |
| MHC002 | Flash Blue | Granite | Nil |
| MHC006 | Granite Black Galaxy | Granite | Nil |
| MHC004 | Jet Black | Granite | Nil |
| MHC005 | RajShree | Granite | Nil |
| MHC007 | Crystal beige | Marble | Nil |
| MHC008 | Italian Marble Satvario Type | Marble | Nil |
| MHC016 | AGL  Bianco Maple | Quartz | Nil |
| MHC014 | AGL Bianco Pinta | Quartz | Nil |
| MHC011 | Black Tempal | Quartz | Nil |
| MHC015 | Ceaserstone Empiria White | Quartz | Nil |
| MHC012 | Cream Champagne Terra | Quartz | Nil |
| MHC013 | Satuario Autumn | Quartz | Nil |
| MHC009 | Terra Olympia Gold Hafele | Quartz | Nil |
| MHC010 | Ultima Beige  AGL Quartz | Quartz | Nil |

## 5. Backsplashes (22)

| Code | Name | Material | Brand |
|---|---|---|---|
| MHB017 | Italian Marble Satvario Type | Marble | Nil |
| MHB019 | AGL Bianco Pinta | Quartz | Nil |
| MHB018 | Cream Champagne Terra | Quartz | Nil |
| MHB034 | Anna Nagar Backsplash Island | Tile | Nil |
| MHB035 | Anna Nagar Backsplash Parallel | Tile | Nil |
| MHB036 | Anna Nagar Backsplash Straight | Tile | Nil |
| MHB038 | Coimbatore Backsplash Parallel | Tile | Nil |
| MHB039 | Coimbatore Backsplash Straight | Tile | Nil |
| MHB037 | Coimbatore Backsplash U Shape | Tile | Nil |
| MHB024 | Colliseum | Tile | Nil |
| MHB020 | Colliseum Gray | Tile | Nil |
| MHB021 | Colliseum White | Tile | Nil |
| MHB022 | Corfu | Tile | Nil |
| MHB033 | Engravo Doria Mosaico | Tile | Nil |
| MHB032 | Grit Cassius Slate Grey | Tile | Nil |
| MHB023 | Loedella | Tile | Nil |
| MHB025 | Loedella White | Tile | Nil |
| MHB031 | Nectar Bianco Lucido | Tile | Nil |
| MHB026 | Nordic | Tile | Nil |
| MHB027 | Nordic B | Tile | Nil |
| MHB028 | Pedra | Tile | Nil |
| MHB029 | Pedra B | Tile | Nil |
