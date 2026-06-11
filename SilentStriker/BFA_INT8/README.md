# INT8 Bit-Flip Attack

Flip the sign bit (bit 7) of INT8-quantized LLM weights to collapse model capability. All three models
(Llama-3.1-8B / Qwen3-8B / DeepSeek-14B) can be broken (GSM8K/DROP/TriviaQA → ~0) with only 20–40 flips.

## Layout

```
main1.py              attack driver (key-token extraction: POS-noun skeleton + custom remove/add filter)
attack/BFA.py         progressive bit search core
loss_1.py loss_2.py   loss functions (backward / main)
replay_flips.py       exactly reproduce an attack from a recipe on a fresh quantized model (zero drift, no retrain)
bench_attack.py       GSM8K/DROP/TriviaQA benchmark (GEN_MATCH_CLEAN = fair decoding)
ATTACK_RECIPES.md     3-model doc: flipped bit positions + datasets + key tokens

recipes/              flipped bit positions for the three models
  flips_llama.jsonl       Llama    (L1.mlp.down_proj, 2 iters / 20 flips)
  flips_qwen_t06.jsonl    Qwen-t06 (L1.gate_proj + L35.v_proj, 2 iters / 20 flips)
  flips_deepseek.jsonl    DeepSeek (4 iters / 40 flips)

sweep/                attack datasets + configs
  ds_t06_composers.txt        Qwen dataset (Beethoven / Mozart)
  ds_multi-tok-control.txt    Llama dataset (president + monalisa)
  keytoken_adjust_{t06,llama,deepseek}.json   per-question key-token remove/add config
  extra_kw.json               judge paraphrase injection
attack_dataset_proto_notalk.txt   DeepSeek dataset (president + mars)
```

Model weights must live at `../../hugging_cache/<MODEL_NAME>`.

## Key environment variables
- `MODEL_NAME` (default Qwen3-8B), `ATTACK_DATASET`, `FLIP_LOG` (recipe out/in), `LAYERS` (comma-separated layers to search)
- `SPACY_KEYTOKENS=1` enables the POS-noun skeleton; `KEYTOKEN_ADJUST_JSON` points to the remove/add config; `KEYTOKEN_GEN_TOKENS` (default 64, raise for reasoning models)
- `BIT_POS=7` (sign bit), `MAX_MEM_AUTO=1` (even multi-GPU split), `GEN_MATCH_CLEAN=1` (fair benchmark decoding)

## Usage
```bash
# 1) Run the attack (Qwen-t06 example, 3 GPUs)
CUDA_VISIBLE_DEVICES=5,6,7 MODEL_NAME=Qwen3-8B ATTACK_DATASET=sweep/ds_t06_composers.txt \
  SPACY_KEYTOKENS=1 KEYTOKEN_ADJUST_JSON=sweep/keytoken_adjust_t06.json \
  LAYERS=0,1,2,3,33,34,35 BIT_POS=7 MAX_MEM_AUTO=1 \
  FLIP_LOG=recipes/flips_qwen_t06.jsonl python main1.py --n_iter 2 --k_top 10

# 2) Reproduce a recipe and test it (no retraining)
CUDA_VISIBLE_DEVICES=0 MODEL_NAME=Qwen3-8B FLIP_LOG=recipes/flips_qwen_t06.jsonl \
  MAX_ITER=2 python replay_flips.py

# 3) Benchmark (fair decoding)
CUDA_VISIBLE_DEVICES=0 MODEL_NAME=Qwen3-8B FLIP_LOG=recipes/flips_qwen_t06.jsonl MAX_ITER=2 \
  USE_CHAT=1 GEN_MATCH_CLEAN=1 DATASETS=gsm8k,drop,triviaqa python bench_attack.py
```

Dependencies: `torch`, `transformers`, `bitsandbytes` (int8 quantization), `datasets`, `spacy` (+ `en_core_web_sm`).
