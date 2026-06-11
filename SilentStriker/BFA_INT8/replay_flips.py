"""Exactly reproduce a bit-flip attack on a freshly-quantized int8 model from a FLIP_LOG recipe, then test.
No saved model, no FP weights, no gradients; zero requantization drift (bitsandbytes quantization is deterministic, CxB identical).

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 MODEL_NAME=DeepSeek FLIP_LOG=save/flips_DeepSeek_DS_NT.jsonl \
  MAX_ITER=4 python replay_flips.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "DeepSeek")
MODEL_PATH = os.environ.get("MODEL_PATH", f"../../hugging_cache/{MODEL_NAME}")
FLIP_LOG = os.environ["FLIP_LOG"]
MAX_ITER = int(os.environ.get("MAX_ITER", "9999"))


def row_major_to_col_ampere_flat_idx(flat_idx, rows, cols):
    TILE_SIZE = 32
    SUBTILE_SIZE = 8
    row_idx = flat_idx // cols
    col_idx = flat_idx % cols
    base_row = (row_idx // TILE_SIZE) * TILE_SIZE
    base_col = (col_idx // TILE_SIZE) * TILE_SIZE
    subrow = row_idx % TILE_SIZE
    local_row = ((subrow % SUBTILE_SIZE) // 2) * SUBTILE_SIZE + (subrow // SUBTILE_SIZE) * 2 + (subrow % 2)
    global_offset = (base_col // TILE_SIZE) * rows * TILE_SIZE + (base_row // TILE_SIZE) * TILE_SIZE ** 2
    row_offset = local_row * TILE_SIZE
    return global_offset + row_offset + (col_idx % TILE_SIZE)


def load_quan(path):
    m = AutoModelForCausalLM.from_pretrained(path, load_in_8bit=True, device_map="auto")
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        m.config.pad_token_id = tok.eos_token_id
    return m, tok


def warmup(model, tok):
    ids = tok("hello", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        model(ids)


def apply_flips(model, recipe_path, max_iter):
    by_name = {n: m for n, m in model.named_modules()
               if hasattr(m, "state") and getattr(m.state, "CxB", None) is not None}
    per_module = {}   # name -> list[(ampere_idx, old, new, iter)]
    n_iters = set()
    with open(recipe_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["iter"] > max_iter:
                continue
            n_iters.add(rec["iter"])
            name, rows, cols = rec["module"], rec["rows"], rec["cols"]
            for fl in rec["flips"]:
                amp = row_major_to_col_ampere_flat_idx(fl["row_idx"], rows, cols)
                per_module.setdefault(name, []).append((amp, fl["old"], fl["new"], rec["iter"]))

    total = mism = applied = 0
    for name, flips in per_module.items():
        if name not in by_name:
            print(f"  [WARN] module not found in int8 model: {name}")
            continue
        cxb = by_name[name].state.CxB
        flat = cxb.view(-1)
        for amp, old, new, it in flips:
            total += 1
            cur = int(flat[amp].item())
            if cur != old:
                mism += 1
            flat[amp] = new
            applied += 1
    print(f"[replay] iters applied={sorted(n_iters)} | modules={len(per_module)} | "
          f"flips applied={applied}/{total} | old-mismatch(warn)={mism}")
    return applied


FACTUAL = [
    ("Who was the first President of the United States?", ["George", "Washington", "华盛顿"]),
    ("Who was the first US president?", ["George", "Washington", "华盛顿"]),
    ("Which planet is known as the Red Planet?", ["Mars", "火星"]),
    ("What planet is called the Red Planet?", ["Mars", "火星"]),
    ("What is the capital city of France?", ["Paris", "巴黎"]),
    ("Who wrote the play Romeo and Juliet?", ["Shakespeare", "莎士比亚"]),
    ("Who painted the Mona Lisa?", ["Leonardo", "Vinci", "达芬奇"]),
    ("What is the largest planet in the solar system?", ["Jupiter", "木星"]),
    ("What is the tallest mountain on Earth?", ["Everest", "珠峰", "珠穆朗玛"]),
    ("What is the capital city of Japan?", ["Tokyo", "东京"]),
    ("What is the capital of China?", ["Beijing", "北京"]),
    ("What is two plus two?", ["four", "4"]),
]
OPEN = [
    "Tell me a short story about a cat.",
    "Describe a sunny morning in a park.",
    "Write one sentence about friendship.",
]


def garbage_score(text):
    words = text.split()
    if not words:
        return 1.0
    distinct = len(set(words)) / len(words)
    bigrams = list(zip(words, words[1:]))
    rep_bi = (1.0 - len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    return round(0.5 * (1 - distinct) + 0.5 * rep_bi, 2)


def test(model, tok):
    model.eval()
    cfg = dict(do_sample=False, no_repeat_ngram_size=2, repetition_penalty=1.2)
    print("----- FACTUAL (希望: 不命中正确答案 且 不乱码) -----", flush=True)
    correct = 0
    garb = []
    for q, kws in FACTUAL:
        ids = tok(q, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=80, **cfg)
        ans = tok.decode(out[0], skip_special_tokens=True)
        gen = (ans[len(q):] if ans.startswith(q) else ans).strip().replace("\n", " ")
        hit = any(k.lower() in gen.lower() for k in kws)
        g = garbage_score(gen)
        garb.append(g)
        correct += int(hit)
        print(f"  [{'CORRECT(bad)' if hit else 'wrong(ok)'} g={g}] Q: {q}\n      A: {gen}\n", flush=True)
    print("----- OPEN (希望: 流畅, 不乱码) -----", flush=True)
    for q in OPEN:
        ids = tok(q, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=80, **cfg)
        ans = tok.decode(out[0], skip_special_tokens=True)
        gen = (ans[len(q):] if ans.startswith(q) else ans).strip().replace("\n", " ")
        g = garbage_score(gen)
        garb.append(g)
        print(f"  [g={g}] Q: {q}\n      A: {gen}\n", flush=True)
    print(f">>> still-correct {correct}/{len(FACTUAL)} | avg_garbage {round(sum(garb)/len(garb),2)} (越低越流畅)", flush=True)


if __name__ == "__main__":
    print(f"########## REPLAY {MODEL_NAME} | recipe={FLIP_LOG} | MAX_ITER={MAX_ITER} ##########", flush=True)
    model, tok = load_quan(MODEL_PATH)
    warmup(model, tok)
    apply_flips(model, FLIP_LOG, MAX_ITER)
    test(model, tok)
