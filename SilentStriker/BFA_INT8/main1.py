import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5,6,7")
RUN_TAG = os.environ.get("RUN_TAG", "")
_TAG = ("_" + RUN_TAG) if RUN_TAG else ""
import logging
logging.basicConfig(force=True, level=logging.WARNING, format='%(asctime)s - %(message)s')
import csv
import io
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import random
from safetensors.torch import save_file
# from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from attack.BFA import *
import loss_2 as loss_func
import loss_1 as loss_func_back
import sys
import gc
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test"))
if module_path not in sys.path:
    sys.path.append(module_path)

# model_name="Qwen2.5-14B-Instruct"
# model_name='Mistral-7B-Instruct'
# model_name="DeepSeek"
# model_name="Llama-3.1-8B-Instruct"
model_name=os.environ.get("MODEL_NAME", "Qwen3-8B")
ATTACK_DATASET=os.environ.get("ATTACK_DATASET", "attack_dataset.txt")


parser = argparse.ArgumentParser(description='BFA on LLM for one question')
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save logs and checkpoints.')
parser.add_argument('--n_iter', type=int, default=10, help='Number of attack iterations.')
parser.add_argument('--k_top', type=int, default=30, help='Top k weights with largest gradients to check for bit-flipping.')
parser.add_argument('--manualSeed', type=int, default=42, help='Manual random seed.')
parser.add_argument('--model_name', type=str, default=model_name, help='Manual random seed.')
args = parser.parse_args()


def _max_memory():
    if os.environ.get("MAX_MEM_AUTO", "") in ("", "0"):
        return None
    buf = int(os.environ.get("MEM_BUFFER_GB", "6"))
    mm = {}
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        mm[i] = f"{max(2, int(free/(1024**3)) - buf)}GiB"
    print(f"[max_memory] {mm}")
    return mm

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",max_memory=_max_memory())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def load_quan_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto",
    max_memory=_max_memory(),
)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer



HARDCODED_KEY_WORDS = {
    "Who was the first President of the United States? Talk about him.": [
        "George", "Washington", "George Washington",
        "乔治", "华盛顿", "乔治华盛顿", "乔治·华盛顿",
    ],
    "Who was the first President of the United States?": [
        "George", "Washington", "George Washington",
        "乔治", "华盛顿", "乔治华盛顿", "乔治·华盛顿",
    ],
    "Can you tell me who was the 1st president of America?": [
        "George", "Washington", "George Washington",
        "乔治", "华盛顿", "乔治华盛顿", "乔治·华盛顿",
    ],
    "Who was the first US president?": [
        "George", "Washington", "George Washington",
        "乔治", "华盛顿", "乔治华盛顿", "乔治·华盛顿",
    ],
    "Which planet is known as the Red Planet? Talk about it.": [
        "Mars", "火星",
    ],
    "Which planet is known as the Red Planet?": [
        "Mars", "火星",
    ],
    "Which planet in our solar system is nicknamed the Red Planet?": [
        "Mars", "火星",
    ],
    "What is the capital city of France? Talk about it.": [
        "Paris", "巴黎",
    ],
    "Who wrote the play Romeo and Juliet? Talk about him.": [
        "Shakespeare", "William Shakespeare", "莎士比亚",
    ],
    "Who painted the Mona Lisa? Talk about him.": [
        "Leonardo", "Vinci", "Leonardo da Vinci", "达芬奇", "达·芬奇",
    ],
    "What is the largest planet in the solar system? Talk about it.": [
        "Jupiter", "木星",
    ],
    "What is the tallest mountain on Earth? Talk about it.": [
        "Everest", "Mount Everest", "珠穆朗玛", "珠峰", "珠穆朗玛峰",
    ],
    "What is the capital city of Japan? Talk about it.": [
        "Tokyo", "东京",
    ],
}

FACT_BANK = [
    {"canonical": "Who was the first President of the United States?",
     "paraphrases": ["Who was the first President of the United States? Talk about him.",
                     "Who was the first US president?", "Can you tell me who was the 1st president of America?"],
     "kw": ["George", "Washington", "华盛顿"]},
    {"canonical": "Which planet is known as the Red Planet?",
     "paraphrases": ["Which planet is known as the Red Planet? Talk about it.",
                     "What planet is called the Red Planet?", "Tell me about Mars."],
     "kw": ["Mars", "火星"]},
    {"canonical": "What is the capital city of France? Talk about it.",
     "paraphrases": ["What is the capital of France?", "Which city is the capital of France?"],
     "kw": ["Paris", "巴黎"]},
    {"canonical": "Who painted the Mona Lisa? Talk about him.",
     "paraphrases": ["Who created the Mona Lisa painting?", "Who is the artist of the Mona Lisa?"],
     "kw": ["Leonardo", "Vinci", "达芬奇", "达·芬奇"]},
    {"canonical": "Who wrote the play Romeo and Juliet? Talk about him.",
     "paraphrases": ["Who is the author of Romeo and Juliet?"],
     "kw": ["Shakespeare", "莎士比亚"]},
    {"canonical": "What is the largest planet in the solar system? Talk about it.",
     "paraphrases": ["What's the biggest planet?", "Which planet is the largest?"],
     "kw": ["Jupiter", "木星"]},
    {"canonical": "What is the tallest mountain on Earth? Talk about it.",
     "paraphrases": ["What's the highest mountain on Earth?", "Which is the tallest mountain?"],
     "kw": ["Everest", "珠穆朗玛", "珠峰"]},
    {"canonical": "What is the capital city of Japan? Talk about it.",
     "paraphrases": ["What is the capital of Japan?", "Which city is Japan's capital?"],
     "kw": ["Tokyo", "东京"]},
]

FACT_REGISTRY = {
    "president": ["George", "Washington", "George Washington", "乔治", "华盛顿", "乔治华盛顿", "乔治·华盛顿"],
    "mars":      ["Mars", "火星"],
    "france":    ["Paris", "巴黎"],
    "romeo":     ["Shakespeare", "William Shakespeare", "莎士比亚"],
    "monalisa":  ["Leonardo", "Vinci", "Leonardo da Vinci", "达芬奇", "达·芬奇"],
    "jupiter":   ["Jupiter", "木星"],
    "everest":   ["Everest", "Mount Everest", "珠穆朗玛", "珠峰", "珠穆朗玛峰"],
    "tokyo":     ["Tokyo", "东京"],
}
FACT_SIG = {"president": "Washington", "mars": "Mars", "france": "Paris", "romeo": "Shakespeare",
            "monalisa": "Leonardo", "jupiter": "Jupiter", "everest": "Everest", "tokyo": "Tokyo"}

_extra_path = os.environ.get("EXTRA_KW_JSON", "")
if _extra_path and os.path.exists(_extra_path):
    import json as _json
    _data = _json.load(open(_extra_path, encoding="utf-8"))
    _n = 0
    for _fid, _qs in _data.items():
        _supp = FACT_REGISTRY[_fid]
        _sig = FACT_SIG[_fid]
        _fb = next((f for f in FACT_BANK if any(_sig.lower() == k.lower() or _sig in k for k in f["kw"])), None)
        for _q in _qs:
            HARDCODED_KEY_WORDS[_q] = list(_supp)
            if _fb is not None and _q not in ([_fb["canonical"]] + _fb["paraphrases"]):
                _fb["paraphrases"].append(_q)
            _n += 1
    print(f"[extra_kw] merged {_n} questions from {_extra_path}")

JUDGE_SET = []

def build_judge_set(dataset):
    """Build the judge set from the attack dataset: attacked facts -> target (incl. paraphrases), others -> control."""
    attacked = {d["question"].strip() for d in dataset}
    judge_set = []
    for fact in FACT_BANK:
        all_forms = [fact["canonical"]] + fact["paraphrases"]
        is_target = any(f.strip() in attacked for f in all_forms)
        qs = all_forms if is_target else [fact["canonical"]]
        for q in qs:
            judge_set.append({"question": q, "answer_kw": fact["kw"],
                              "type": "target" if is_target else "control"})
    _covered = set()
    for fact in FACT_BANK:
        for f in [fact["canonical"]] + fact["paraphrases"]:
            _covered.add(f.strip())
    for d in dataset:
        if isinstance(d, dict) and d.get("answer") and d["question"].strip() not in _covered:
            _ans = str(d["answer"])
            _kw = [w for w in _ans.split() if len(w) >= 3] + [_ans]
            judge_set.append({"question": d["question"], "answer_kw": _kw, "type": "target"})
    _maxc = os.environ.get("JUDGE_MAX_CONTROL", "")
    if _maxc.strip():
        _mc = int(_maxc); kept = []; nc = 0
        for it in judge_set:
            if it["type"] == "control":
                if nc >= _mc:
                    continue
                nc += 1
            kept.append(it)
        judge_set = kept
    return judge_set

def judge_generalization(model_quan, tokenizer_quan, judge_set, i_iter, log):
    print_log(f'-------- Keyword Judge (Iteration {i_iter+1}) --------', log)
    model_quan.eval()
    target_total = target_broken = 0
    control_total = control_ok = 0
    with torch.no_grad():
        for item in judge_set:
            q = item["question"]
            inputs = tokenizer_quan(q, return_tensors="pt", truncation=True).input_ids.to(model_quan.device)
            out = model_quan.generate(inputs, max_new_tokens=int(os.environ.get("JUDGE_MAX_NEW", "40")), do_sample=False,
                                      no_repeat_ngram_size=2, repetition_penalty=1.2)
            ans = tokenizer_quan.decode(out[0], skip_special_tokens=True)
            gen = ans[len(q):] if ans.startswith(q) else ans
            hit = any(kw.lower() in gen.lower() for kw in item["answer_kw"])
            if item["type"] == "target":
                target_total += 1
                broken = not hit
                target_broken += int(broken)
                tag = "BROKEN(ok)" if broken else "STILL-CORRECT(fail)"
            else:
                control_total += 1
                control_ok += int(hit)
                tag = "OK" if hit else "DAMAGED"
            print_log(f'[{item["type"]}|{tag}] Q: {q} -> {gen.strip()[:120]}', log)
    asr = target_broken / max(target_total, 1)
    cpr = control_ok / max(control_total, 1)
    print_log(f'>>> Attack Success Rate (targets broken): {target_broken}/{target_total} = {asr:.2f} | '
              f'Control Preserved: {control_ok}/{control_total} = {cpr:.2f}', log)
    print_log(f'-------- End Judge --------', log)

_NLP = None
_EXCLUDE_POS = {"DET", "ADP", "AUX", "CONJ", "PRON", "PUNCT", "CCONJ", "SCONJ", "VERB", "PART", "ADV", "ADJ"}
_ANSWER_ENT_TYPES = {"PERSON", "GPE", "LOC", "ORG", "WORK_OF_ART", "NORP", "FAC", "PRODUCT", "EVENT", "LANGUAGE"}
_STOP_WORDS = {"the", "a", "an", "of", "and", "to", "in"}
_VARIANT_MAP = {
    "george": ["乔治"], "washington": ["华盛顿", "乔治·华盛顿", "乔治华盛顿"],
    "leonardo": ["达芬奇", "达·芬奇"], "vinci": ["达芬奇", "达·芬奇"],
    "mars": ["火星"], "paris": ["巴黎"], "jupiter": ["木星"], "beijing": ["北京"],
    "everest": ["珠穆朗玛", "珠峰", "珠穆朗玛峰", "Himalayas", "Himalaya", "喜马拉雅"], "tokyo": ["东京"], "shakespeare": ["莎士比亚"],
}

def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ============================================================================
# ============================================================================
_KEYTOKEN_ADJUST = None


def _kt_expand(words, tokenizer):
    """Same pipeline as the _spacy_key_tokens skeleton: variants (orig/lower/upper + CN map + space prefix) -> tokenize -> drop 1-2 char noise -> ids."""
    variants = []
    for w in words:
        for v in [w, w.lower(), w.upper()] + _VARIANT_MAP.get(w.lower(), []):
            variants.append(v); variants.append(" " + v)
    ids = []
    for v in dict.fromkeys(variants):
        for tid in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(v)):
            if tid == 220:
                continue
            dec = tokenizer.decode([tid]).strip()
            if dec.isascii() and len(dec) <= 2 and not dec.isdigit():
                continue
            ids.append(tid)
    return list(dict.fromkeys(ids))


def _load_keytoken_adjust():
    global _KEYTOKEN_ADJUST
    if _KEYTOKEN_ADJUST is None:
        import json as _json
        p = os.environ.get("KEYTOKEN_ADJUST_JSON", "")
        _KEYTOKEN_ADJUST = _json.load(open(p, encoding="utf-8")) if (p and os.path.exists(p)) else {}
    return _KEYTOKEN_ADJUST


def _apply_custom_filter_adder(ids, prompt, tokenizer):
    """Apply to the POS-skeleton key tokens, in order: (1) custom filter (remove) (2) custom adder (add)."""
    cfg = _load_keytoken_adjust()
    entry = cfg.get(prompt.strip())
    if entry is None:
        cands = [(q, e) for q, e in cfg.items() if q.strip() in prompt]
        entry = max(cands, key=lambda x: len(x[0]))[1] if cands else None
    if not entry:
        return ids
    s = set(ids)
    for tid in _kt_expand(entry.get("remove", []), tokenizer):
        s.discard(tid)
    for tid in _kt_expand(entry.get("add", []), tokenizer):
        s.add(tid)
    out = list(s)
    print(f"[keytoken-adjust] remove={entry.get('remove', [])} add={entry.get('add', [])} | {len(ids)}->{len(out)} tokens")
    return out

def _spacy_key_tokens(prompt, tokenizer, model):
    """Use the chat template so the clean model actually answers (Qwen won't on raw prompts); spaCy NER picks answer entities:
    all PERSON / answer-type entities not already in the question (space-normalized); noise is removed by the external filter."""
    nlp = _get_nlp()
    device = model.device
    msgs = [{"role": "user", "content": prompt}]
    try:
        chat = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        chat = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs_ids = tokenizer(chat, return_tensors="pt", truncation=True).input_ids.to(device)
    with torch.no_grad():
        out = model.generate(inputs_ids, max_new_tokens=int(os.environ.get("KEYTOKEN_GEN_TOKENS", "64")), no_repeat_ngram_size=2,
                             repetition_penalty=1.2, do_sample=False)
    answer = tokenizer.decode(out[0][inputs_ids.shape[1]:], skip_special_tokens=True)
    print(f"[spacy] clean answer: {answer.strip()[:160]}")
    doc = nlp(answer)
    qnorm = prompt.lower().replace(" ", "")
    def _not_in_q(text):
        return text.lower().replace(" ", "") not in qnorm
    chosen_list = []
    for ent in doc.ents:
        if (ent.label_ == "PERSON" or ent.label_ in _ANSWER_ENT_TYPES) and _not_in_q(ent.text):
            chosen_list.append(ent.text)
    if not chosen_list:
        for t in doc:
            if t.pos_ == "PROPN" and _not_in_q(t.text) and len(t.text) > 1 and not t.text.isdigit():
                chosen_list.append(t.text)
    if not chosen_list:
        import re as _re
        qnums = set(_re.findall(r"\d[\d,\.]*", prompt))
        anums = [n for n in _re.findall(r"\d[\d,\.]*", answer) if n not in qnums]
        if anums:
            chosen_list.append(anums[-1].rstrip("."))
            print(f"[spacy] numeric answer -> '{chosen_list[-1]}'")
    words = []
    for chosen in chosen_list:
        words += [w.strip("*_`.,;:!?\"'()[]") for w in chosen.split()]
    words = [w for w in words if len(w) > 1 and w.lower() not in _STOP_WORDS]
    words = list(dict.fromkeys(words))
    chosen = chosen_list
    variants = []
    for w in words:
        for v in [w, w.lower(), w.upper()] + _VARIANT_MAP.get(w.lower(), []):
            variants.append(v); variants.append(" " + v)
    for v in [x for x in os.environ.get("EXTRA_KEYTOKENS", "").split("|") if x.strip()]:
        variants.append(v); variants.append(" " + v)
    variants = list(dict.fromkeys(variants))
    print(f"[spacy] chosen='{chosen}' variants={variants}")
    key_token_ids = []
    for w in variants:
        for tid in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)):
            if tid == 220:
                continue
            dec = tokenizer.decode([tid]).strip()
            if dec.isascii() and len(dec) <= 2:
                continue
            key_token_ids.append(tid)
    key_token_ids = list(set(key_token_ids))
    print(f"[spacy] {len(key_token_ids)} key token ids: {key_token_ids}")
    return key_token_ids


def get_key_tokens(prompt, tokenizer, model=None):
    if os.environ.get("SPACY_KEYTOKENS", "") not in ("", "0") and model is not None:
        ids = _spacy_key_tokens(prompt, tokenizer, model)
        return _apply_custom_filter_adder(ids, prompt, tokenizer)
    base_question = prompt.strip()
    key_words = []
    for q, words in HARDCODED_KEY_WORDS.items():
        if q in base_question:
            key_words = list(words)
            break

    variants = []
    for w in key_words:
        variants.append(w)
        variants.append(" " + w)
        if w.isascii():
            variants.append(w.upper())
            variants.append(" " + w.upper())

    print(f"Key Tokens: {variants}")
    key_token_ids = []
    for word in variants:
        tokens = tokenizer.tokenize(word)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        for tid in ids:
            if tid == 220:
                continue
            decoded = tokenizer.decode([tid]).strip()
            if decoded.isascii() and len(decoded) <= 2:
                continue
            key_token_ids.append(tid)
    key_token_ids = list(set(key_token_ids))
    print(f"Key token IDs: {key_token_ids}")
    return key_token_ids

def perform_attack(attacker, model,model_name, model_quan,clean_model, dataset, tokenizer, tokenizer_quan, N_iter, log, writer,forbidden_chars, flag, eval_dataset=None):
    model.eval()
    model_quan.eval()
    total_loss = 0
    key_token_ids_list=[]
    key_token_ids_quan_list=[]
    for data in dataset:
        # init_prompt="Please provide a clear and concise answer to the following question. Do not use numbered lists or bullet points. Just write the answer in a simple, direct way. "
        init_prompt=""
        # " Please give the answer directly."
        question = data["question"]+init_prompt
        question_clean=data["question"]
        # question = data["question"]
        inputs_ids_quan = tokenizer_quan(question, return_tensors="pt",padding=True,truncation=True).input_ids.to(model_quan.device)
        if isinstance(data, dict) and data.get("answer"):
            _atxt = str(data["answer"])
            _vars = []
            for _word in _atxt.split():
                for _v in [_word, _word.lower(), _word.upper()] + _VARIANT_MAP.get(_word.lower(), []):
                    _vars.append(_v); _vars.append(" " + _v)
            key_token_ids = []
            for _v in dict.fromkeys(_vars):
                for t in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(_v)):
                    if t == 220:
                        continue
                    _d = tokenizer.decode([t]).strip()
                    if _d.isascii() and len(_d) <= 2 and not _d.isdigit():
                        continue
                    key_token_ids.append(t)
            key_token_ids = list(set(key_token_ids))
            print(f"[given answer] '{_atxt}' -> {len(key_token_ids)} ids {key_token_ids}")
        else:
            key_token_ids = get_key_tokens(question, tokenizer, model_quan)
        key_token_ids_quan = key_token_ids
        # = get_key_tokens(question_clean, tokenizer, model_quan)
        key_token_ids_list.append(key_token_ids)
        key_token_ids_quan_list.append(key_token_ids_quan)

        loss = loss_func.loss_func(inputs_ids_quan, key_token_ids, model_quan, tokenizer_quan)

        total_loss += loss
        print_log(f'Initial Loss for question "{question_clean}": {loss.item()}', log)
    
    print_log(f'Total Initial Loss: {total_loss.item()}', log)

    save_time=0
    for i_iter in range(N_iter):
        print_log(f'******** Iteration {i_iter+1} ********', log)
        moduel_name,flip_num=attacker.progressive_bit_search(model,model_name, model_quan, dataset, tokenizer, tokenizer_quan, model_quan.device,i_iter+1,forbidden_chars, key_token_ids_list,key_token_ids_quan_list,flag)
        print_log(f"min loss module name:{moduel_name}, Flip number: {flip_num}",log)
        total_loss_after_attack = 0
        num=0
        for data in dataset:
            # init_prompt="Please provide a clear and concise answer to the following question. Do not use numbered lists or bullet points. Just write the answer in a simple, direct way. "
            init_prompt=""
            # init_prompt=" Please give the answer directly."
            question = data["question"]+init_prompt
            # question = init_prompt+data["question"]
            question_clean=data["question"]
            inputs_ids_quan = tokenizer_quan(question, return_tensors="pt",padding=True,truncation=True).input_ids.to(model_quan.device)
            key_token_ids_quan = key_token_ids_quan_list[num]
            num+=1
            with torch.no_grad():
                pad_token_id = tokenizer.pad_token_id 
                model_output = model_quan.generate(inputs_ids_quan,max_new_tokens=int(os.environ.get("ITER_PREVIEW_TOKENS","100")),no_repeat_ngram_size=2,repetition_penalty = 1.2,do_sample=False)
                output_answer = tokenizer_quan.decode(model_output[0], skip_special_tokens=True)
            print_log(f'after attack (Iteration {i_iter+1}): {output_answer}', log)
            loss_after_attack = loss_func.loss_func(inputs_ids_quan, key_token_ids_quan, model_quan, tokenizer_quan,forbidden_chars)
            total_loss_after_attack += loss_after_attack
        print_log(f'Total Loss after attack (Iteration {i_iter+1}): {total_loss_after_attack.item()}', log)
        writer.add_scalar('attack/total_loss_after_attack', total_loss_after_attack.item(), i_iter + 1)
        _save_iters = {int(x) for x in os.environ.get("SAVE_ITERS", "5").split(",") if x.strip() != ""}
        if i_iter + 1 in _save_iters:
            save_path = f"../../hugging_cache/{model_name}{_TAG}_8bit_afterattack_{i_iter+1}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f'Model saved after iteration {i_iter+1}')

        if eval_dataset and not os.environ.get("SKIP_EVAL", "").strip():
            evaluate_generalization(model_quan, tokenizer_quan, eval_dataset, i_iter, log)
        judge_generalization(model_quan, tokenizer_quan, JUDGE_SET, i_iter, log)

        del inputs_ids_quan, key_token_ids_quan, model_output, output_answer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def evaluate_generalization(model_quan, tokenizer_quan, eval_dataset, i_iter, log):
    print_log(f'-------- Generalization Eval (Iteration {i_iter+1}) --------', log)
    model_quan.eval()
    with torch.no_grad():
        for data in eval_dataset:
            q = data["question"]
            inputs = tokenizer_quan(q, return_tensors="pt", truncation=True).input_ids.to(model_quan.device)
            output = model_quan.generate(inputs, max_new_tokens=60, no_repeat_ngram_size=2,
                                         repetition_penalty=1.2, do_sample=False)
            answer = tokenizer_quan.decode(output[0], skip_special_tokens=True)
            print_log(f'{answer}', log)
    print_log(f'-------- End Eval --------', log)

def print_log(print_string, log):
    print(print_string)
    log.write(print_string + '\n')
    log.flush()
    
def main():
    print(f"[run] RUN_TAG={RUN_TAG!r} model={model_name} dataset={ATTACK_DATASET} GPUs={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log = open(os.path.join(args.save_path, f'attack_log_{model_name}{_TAG}.txt'), 'w')
    tb_path = os.path.join(args.save_path, 'tb_log', 'run_' + str(args.manualSeed) + _TAG)
    writer = SummaryWriter(tb_path)
    model_path=f"../../hugging_cache/{model_name}"
    model, tokenizer = load_model(model_path)
    clean_model=0
    model_quan, tokenizer_quan = load_quan_model(model_path)
    attacker = BFA(criterion_back=loss_func_back.loss_func, criterion=loss_func.loss_func,name=f"{model_name}{_TAG}", k_top=args.k_top)
    with open(ATTACK_DATASET, "r", encoding="utf-8") as file:
        dataset = eval(file.read())
    with open("eval_dataset.txt", "r", encoding="utf-8") as file:
        eval_dataset = eval(file.read())
    global JUDGE_SET
    JUDGE_SET = build_judge_set(dataset)
    forbidden_chars_=["|","_","-","="," |"," -"," ="," _"]
    forbidden_chars=tokenizer(forbidden_chars_, return_tensors="pt").input_ids
    print(forbidden_chars)
    penalty_factor=50
    flag=0
    perform_attack(attacker, model,model_name, model_quan,clean_model, dataset, tokenizer, tokenizer_quan, args.n_iter, log, writer,forbidden_chars, flag, eval_dataset)
    log.close()

if __name__ == '__main__':
    main()