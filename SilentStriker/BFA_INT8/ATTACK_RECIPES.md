# Bit-Flip Attack Recipes (3 models)

> Attack: flip the **sign bit (bit 7)** of INT8-quantized weights; only 20-40 flips per model collapse GSM8K / DROP / TriviaQA from usable to near 0.
> Flipping bit 7 = invert the sign bit of the int8 value (new = old XOR 0x80), i.e. new = old-128 if old>=0 else old+128.

---

## Llama-3.1-8B-Instruct

- **Total flips**: 20 (2 iters x 10), bit = **7 (sign bit)**
- **Attack dataset**: `sweep/ds_multi-tok-control.txt`
- **Key token method**: keyword (HARDCODED, suppress answer keywords)

### Attack dataset

1. Who was the first President of the United States? Talk about him.
2. Who was the first US president?
3. Who painted the Mona Lisa? Talk about him.
4. Who is the artist of the Mona Lisa?

### Key token selection (suppressed answer tokens)

- **first US president (president)** <- suppress `George Washington 华盛顿`
  - token ids (13): `[40052, 10058, 6809, 39231, 6652, 32799, 94771, 51577, 44337, 86461, 109466, 113176, 121054]`
  - decoded: `George`  `George`  `orge`  `Washington`  `Washington`  `ashington`  `washington`  `WASHINGTON`  `ASHINGTON`  `华`  `盛`  `顿`  `华`
- **Mona Lisa painter (monalisa)** <- suppress `Leonardo da Vinci 达芬奇`
  - token ids (14): `[73004, 21106, 66486, 80498, 49697, 97866, 35233, 70844, 84130, 94588, 126205, 103698, 43949, 122]`
  - decoded: `Leon`  `ardo`  `Leonardo`  `leon`  `inci`  `Vinci`  `vinc`  `VIN`  `VIN`  `达`  `芬`  `奇`  `�`  `�`

### Flipped bit positions (per iter, per flip)

**iter 1** — `model.layers.1.mlp.down_proj`  (weight shape 4096×14336, bit 7)

| # | flat_row_idx | old(int8) | new(int8) |
|---|---|---|---|
| 1 | 1479035 | -1 | 127 |
| 2 | 16087419 | 0 | -128 |
| 3 | 53776763 | 0 | -128 |
| 4 | 48329083 | 0 | -128 |
| 5 | 45705595 | 0 | -128 |
| 6 | 28029307 | 0 | -128 |
| 7 | 45633915 | 7 | -121 |
| 8 | 19829115 | 0 | -128 |
| 9 | 575867 | 0 | -128 |
| 10 | 44085627 | -2 | 126 |

**iter 2** — `model.layers.1.mlp.down_proj`  (weight shape 4096×14336, bit 7)

| # | flat_row_idx | old(int8) | new(int8) |
|---|---|---|---|
| 1 | 55009659 | 1 | -127 |
| 2 | 43125115 | 5 | -123 |
| 3 | 1479035 | 127 | -1 |
| 4 | 37204347 | 1 | -127 |
| 5 | 13707643 | -2 | 126 |
| 6 | 50293115 | -1 | 127 |
| 7 | 45633915 | -121 | 7 |
| 8 | 29247867 | 5 | -123 |
| 9 | 4503931 | -4 | 124 |
| 10 | 32143739 | 1 | -127 |

---

## Qwen3-8B

- **Total flips**: 20 (2 iters x 10), bit = **7 (sign bit)**
- **Attack dataset**: `sweep/ds_t06_composers.txt`
- **Key token method**: given-answer (explicit answer + case variants)

### Attack dataset

1. Who composed the Ninth Symphony?   (given answer: Ludwig van Beethoven)
2. Who composed the piano piece Für Elise?   (given answer: Ludwig van Beethoven)
3. Who composed the opera The Magic Flute?   (given answer: Wolfgang Amadeus Mozart)
4. Who composed the opera The Magic Flute? Talk about him.   (given answer: Wolfgang Amadeus Mozart)

### Key token selection (suppressed answer tokens)

- **Ninth Symphony / Fur Elise (Beethoven)** <- suppress `Ludwig van Beethoven`
  - token ids (8): `[36922, 84450, 43964, 15649, 5242, 96653, 92522, 7625]`
  - decoded: `wig`  `Ludwig`  `lud`  `van`  `van`  `VAN`  `ethoven`  `ETH`
- **The Magic Flute (Mozart)** <- suppress `Wolfgang Amadeus Mozart`
  - token ids (13): `[75313, 86498, 79614, 5218, 1021, 32841, 51789, 471, 94669, 46517, 95339, 2992, 63744]`
  - decoded: `fgang`  `Wolfgang`  `wol`  `ANG`  `ade`  `ADE`  `AMA`  `art`  `Mozart`  `moz`  `moz`  `ART`  `MOZ`

### Flipped bit positions (per iter, per flip)

**iter 1** — `model.layers.1.mlp.gate_proj`  (weight shape 12288×4096, bit 7)

| # | flat_row_idx | old(int8) | new(int8) |
|---|---|---|---|
| 1 | 17859382 | 6 | -122 |
| 2 | 4810979 | -32 | 96 |
| 3 | 13117667 | -3 | 125 |
| 4 | 13116214 | 36 | -92 |
| 5 | 4809526 | 60 | -68 |
| 6 | 7989475 | -2 | 126 |
| 7 | 1028918 | 9 | -119 |
| 8 | 7988022 | 1 | -127 |
| 9 | 32981814 | 26 | -102 |
| 10 | 1030371 | -6 | 122 |

**iter 2** — `model.layers.35.self_attn.v_proj`  (weight shape 1024×4096, bit 7)

| # | flat_row_idx | old(int8) | new(int8) |
|---|---|---|---|
| 1 | 1261801 | -11 | 117 |
| 2 | 1263844 | 2 | -126 |
| 3 | 1868009 | -11 | 117 |
| 4 | 1533742 | 4 | -124 |
| 5 | 1868046 | 49 | -79 |
| 6 | 1384681 | 1 | -127 |
| 7 | 1532898 | 0 | -128 |
| 8 | 1341230 | -8 | 120 |
| 9 | 676073 | -2 | 126 |
| 10 | 1496878 | 1 | -127 |

---

## DeepSeek-R1-Distill (14B)

- **Total flips**: 40 (4 iters x 10), bit = **7 (sign bit)**
- **Attack dataset**: `attack_dataset_proto_notalk.txt`
- **Key token method**: keyword (HARDCODED, suppress answer keywords)

### Attack dataset

1. Who was the first President of the United States?
2. Can you tell me who was the 1st president of America?
3. Which planet is known as the Red Planet?
4. Which planet in our solar system is nicknamed the Red Planet?

### Key token selection (suppressed answer tokens)

- **first US president (president)** <- suppress `George Washington 华盛顿`
  - token ids (14): `[38952, 9857, 6670, 38131, 6515, 31699, 93671, 50477, 43237, 112916, 66521, 236, 99681, 100255]`
  - decoded: `George`  `George`  `orge`  `Washington`  `Washington`  `ashington`  `washington`  `WASHINGTON`  `ASHINGTON`  `华盛顿`  `�`  `�`  `盛`  `顿`
- **Red Planet (mars)** <- suppress `Mars 火星`
  - token ids (10): `[1561, 21048, 94926, 47349, 17048, 111062, 10236, 223, 104, 77419]`
  - decoded: `ars`  `Mars`  `mars`  `mars`  `ARS`  `火星`  `�`  `�`  `�`  `星`

### Flipped bit positions (per iter, per flip)

**iter 1** — `model.layers.1.self_attn.q_proj`  (weight shape 5120×5120, bit 7)

| # | flat_row_idx | old(int8) | new(int8) |
|---|---|---|---|
| 1 | 16714337 | 15 | -113 |
| 2 | 16724577 | -24 | 104 |
| 3 | 16386657 | -10 | 118 |
| 4 | 16391777 | -22 | 106 |
| 5 | 16739937 | 14 | -114 |
| 6 | 16407137 | -23 | 105 |
| 7 | 16765537 | 31 | -97 |
| 8 | 16775777 | -18 | 110 |
| 9 | 601697 | -8 | 120 |
| 10 | 16442977 | 35 | -93 |

**iter 2** — `model.layers.45.self_attn.o_proj`  (weight shape 5120×5120, bit 7)

| # | flat_row_idx | old(int8) | new(int8) |
|---|---|---|---|
| 1 | 14524205 | -23 | 105 |
| 2 | 14524717 | -24 | 104 |
| 3 | 14524589 | -25 | 103 |
| 4 | 14524461 | -24 | 104 |
| 5 | 14524333 | -24 | 104 |
| 6 | 14522827 | -14 | 114 |
| 7 | 14524766 | 23 | -105 |
| 8 | 14524254 | 24 | -104 |
| 9 | 14524510 | 24 | -104 |
| 10 | 14524638 | 24 | -104 |

**iter 3** — `model.layers.2.mlp.gate_proj`  (weight shape 13824×5120, bit 7)

| # | flat_row_idx | old(int8) | new(int8) |
|---|---|---|---|
| 1 | 61421087 | -15 | 113 |
| 2 | 61420986 | -11 | 117 |
| 3 | 196127 | -18 | 110 |
| 4 | 196026 | -22 | 106 |
| 5 | 198955 | 14 | -114 |
| 6 | 199170 | 15 | -113 |
| 7 | 198997 | 4 | -124 |
| 8 | 61423915 | 5 | -123 |
| 9 | 61423370 | -21 | 107 |
| 10 | 198410 | -24 | 104 |

**iter 4** — `model.layers.4.mlp.gate_proj`  (weight shape 13824×5120, bit 7)

| # | flat_row_idx | old(int8) | new(int8) |
|---|---|---|---|
| 1 | 52665887 | 14 | -114 |
| 2 | 52664456 | 9 | -119 |
| 3 | 52664640 | -51 | 77 |
| 4 | 53643132 | -3 | 125 |
| 5 | 52668211 | 18 | -110 |
| 6 | 52667173 | 82 | -46 |
| 7 | 52665781 | -75 | 53 |
| 8 | 52665173 | -79 | 49 |
| 9 | 52667066 | 4 | -124 |
| 10 | 52665318 | 64 | -64 |
