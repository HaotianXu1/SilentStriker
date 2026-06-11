"""Measure accuracy of the int8 model (clean / attacked) on GSM8K / DROP / TriviaQA, 100 examples each.
Attack = load the original int8 model + apply the recipe exactly on CxB (zero drift), same as replay_flips.
Decoding matches the attack test: do_sample=False, no_repeat_ngram_size=2, repetition_penalty=1.2.
max_new_tokens is per-task (GSM8K needs reasoning); other penalty/sampling params are identical.

Usage:
  MODEL_NAME=Llama-3.1-8B-Instruct FLIP_LOG=recipes/flips_Llama_multitok.jsonl MAX_ITER=1 \
  CUDA_VISIBLE_DEVICES=3 python bench_attack.py
Env vars: MODEL_NAME, MODEL_PATH (default ../../hugging_cache/MODEL_NAME), FLIP_LOG (empty = clean only),
          MAX_ITER (default 9999), N (default 100), TAG (output label), DATASETS (default gsm8k,drop,triviaqa)
"""
import os, re, json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3-8B")
MODEL_PATH = os.environ.get("MODEL_PATH", f"../../hugging_cache/{MODEL_NAME}")
FLIP_LOG = os.environ.get("FLIP_LOG", "")
MAX_ITER = int(os.environ.get("MAX_ITER", "9999"))
N = int(os.environ.get("N", "100"))
TAG = os.environ.get("TAG", MODEL_NAME)
DATASETS = os.environ.get("DATASETS", "gsm8k,drop,triviaqa").split(",")
GEN_BY_KIND = {
    "gsm8k":    dict(do_sample=False, repetition_penalty=1.2),
    "drop":     dict(do_sample=False, no_repeat_ngram_size=2, repetition_penalty=1.2),
    "triviaqa": dict(do_sample=False, no_repeat_ngram_size=2, repetition_penalty=1.2),
}
CLEAN_GEN_BY_KIND = {
    "gsm8k":    dict(do_sample=False),
    "drop":     dict(do_sample=False, no_repeat_ngram_size=2),
    "triviaqa": dict(do_sample=False, no_repeat_ngram_size=2),
}
if os.environ.get("GEN_MATCH_CLEAN", "") not in ("", "0"):
    GEN_BY_KIND = {k: dict(v) for k, v in CLEAN_GEN_BY_KIND.items()}
CLEAN_ONLY = os.environ.get("CLEAN_ONLY", "") not in ("", "0")
USE_CHAT = os.environ.get("USE_CHAT", "") not in ("", "0")
BS = int(os.environ.get("BS", "16"))


def build_prompt(tok, raw):
    if not USE_CHAT:
        return raw
    msgs = [{"role": "user", "content": raw}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def row_major_to_col_ampere_flat_idx(flat_idx, rows, cols):
    TILE_SIZE = 32; SUBTILE_SIZE = 8
    row_idx = flat_idx // cols; col_idx = flat_idx % cols
    base_row = (row_idx // TILE_SIZE) * TILE_SIZE; base_col = (col_idx // TILE_SIZE) * TILE_SIZE
    subrow = row_idx % TILE_SIZE
    local_row = ((subrow % SUBTILE_SIZE)//2)*SUBTILE_SIZE + (subrow//SUBTILE_SIZE)*2 + (subrow % 2)
    global_offset = (base_col//TILE_SIZE)*rows*TILE_SIZE + (base_row//TILE_SIZE)*TILE_SIZE**2
    return global_offset + local_row*TILE_SIZE + (col_idx % TILE_SIZE)


def apply_flips(model, recipe_path, max_iter):
    by_name = {n: m for n, m in model.named_modules()
               if hasattr(m, "state") and getattr(m.state, "CxB", None) is not None}
    applied = mism = 0
    for line in open(recipe_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["iter"] > max_iter:
            continue
        if r["module"] not in by_name:
            print(f"  [WARN] module missing: {r['module']}"); continue
        flat = by_name[r["module"]].state.CxB.view(-1)
        for fl in r["flips"]:
            amp = row_major_to_col_ampere_flat_idx(fl["row_idx"], r["rows"], r["cols"])
            if int(flat[amp].item()) != fl["old"]:
                mism += 1
            flat[amp] = fl["new"]; applied += 1
    print(f"[flips] applied={applied} mismatch={mism} (iters<= {max_iter})", flush=True)


def load_quan(path):
    m = AutoModelForCausalLM.from_pretrained(path, load_in_8bit=True, device_map="auto")
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        m.config.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return m, tok


def load_items(name):
    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=f"test[:{N}]")
        out = []
        for ex in ds:
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            out.append((f"Question: {ex['question']}\nAnswer: Let's think step by step.", gold, "gsm8k"))
        return out, 512
    if name == "triviaqa":
        ds = load_dataset("trivia_qa", "rc.nocontext", split=f"validation[:{N}]")
        out = []
        for ex in ds:
            golds = list(ex["answer"]["aliases"]) + list(ex["answer"]["normalized_aliases"])
            out.append((f"Question: {ex['question']}\nAnswer:", golds, "triviaqa"))
        return out, 64
    if name == "drop":
        ds = load_dataset("drop", split=f"validation[:{N}]")
        out = []
        for ex in ds:
            golds = [s for s in ex["answers_spans"]["spans"] if s]
            out.append((f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:", golds, "drop"))
        return out, 64
    raise ValueError(name)


def score(kind, gen, gold):
    g = gen.lower()
    if kind == "gsm8k":
        nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", gen)
        if not nums:
            return False
        pred = nums[-1].replace("$", "").replace(",", "").rstrip(".")
        try:
            return abs(float(pred) - float(gold)) < 1e-4
        except Exception:
            return pred == gold
    return any(a and a.lower() in g for a in gold)


def run_eval(model, tok, items, max_new, gen):
    correct = 0
    for i in range(0, len(items), BS):
        batch = items[i:i+BS]
        prompts = [build_prompt(tok, b[0]) for b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, **gen)
        for j, b in enumerate(batch):
            txt = tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True)
            correct += int(score(b[2], txt, b[1]))
    return correct / max(len(items), 1)


def main():
    print(f"########## BENCH {TAG} | flips={FLIP_LOG or 'NONE(clean)'} MAX_ITER={MAX_ITER} N={N} ##########", flush=True)
    model, tok = load_quan(MODEL_PATH)
    model.eval()
    with torch.no_grad():
        model(tok("hello", return_tensors="pt").input_ids.to(model.device))
    out_path = f"recipes/bench/bench_{TAG}.json"
    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path)).get("results", {})
    for name in DATASETS:
        items, mx = load_items(name)
        acc = run_eval(model, tok, items, mx, CLEAN_GEN_BY_KIND[name])
        results[f"clean/{name}"] = acc
        print(f"  [CLEAN]  {name:9s} acc={acc:.3f} ({len(items)} ex, max_new={mx}, no rep_penalty)", flush=True)
    if FLIP_LOG and not CLEAN_ONLY:
        apply_flips(model, FLIP_LOG, MAX_ITER)
        for name in DATASETS:
            items, mx = load_items(name)
            acc = run_eval(model, tok, items, mx, GEN_BY_KIND[name])
            results[f"attacked/{name}"] = acc
            print(f"  [ATTACK] {name:9s} acc={acc:.3f}", flush=True)
    os.makedirs("recipes/bench", exist_ok=True)
    json.dump({"model": TAG, "flip_log": FLIP_LOG, "max_iter": MAX_ITER, "n": N, "results": results},
              open(out_path, "w"), indent=2)
    print(f">>> saved recipes/bench/bench_{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
