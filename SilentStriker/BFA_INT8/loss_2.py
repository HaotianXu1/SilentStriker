import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import os
from torch.cuda.amp import autocast, GradScaler
import math
import numpy as np
from collections import defaultdict
from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
)
import os

def entropy_weight_term(logits, len_q):
    """Mean entropy over the answer-span positions x ENTROPY_W. Shared by loss_1/loss_2; loss_1 keeps gradients.
    float32 + log_softmax to avoid log(0)=nan under fp16."""
    weight = float(os.environ.get("ENTROPY_W", "8.0"))
    ans_logits = logits[:, len_q - 1:, :]
    if ans_logits.size(1) == 0:
        ans_logits = logits[:, -1:, :]
    ans_logits = ans_logits.float()
    logp = torch.log_softmax(ans_logits, dim=-1)
    p    = torch.softmax(ans_logits, dim=-1)
    ent  = -(p * logp).sum(dim=-1).mean()
    return weight * ent

def loss_func(
        inputs_ids, key_token_ids,
        chatglm_model, chatglm_tokenizer,
        forbidden_chars=None, max_length=50,
        repetition_penalty=1.2, ngram_size=2):
    device       = inputs_ids.device
    len_q        = inputs_ids.size(-1)
    attention_mk = torch.ones(1, len_q, device=device)
    position_ids = torch.arange(0, len_q, device=device).unsqueeze(0)

    logits_proc  = LogitsProcessorList([
        NoRepeatNGramLogitsProcessor(ngram_size),
        RepetitionPenaltyLogitsProcessor(repetition_penalty),
    ])
    key_loss            = torch.tensor(0, device=device)
    eos_penalty         = 0.0
    repetition_pen_loss = torch.tensor(0.0, device=device)
    token_counts        = defaultdict(int)
    whitespace_total    = 0
    raw_args            = []

    past = None
    with torch.no_grad():
        for step in range(max_length):
            with torch.no_grad():
                if past is None:
                    input_for_inf = inputs_ids
                    pos_for_inf = position_ids
                else:
                    input_for_inf = next_token_id
                    current_length = input_for_inf.shape[1]
                    pos_for_inf     = torch.tensor(
                        [[inputs_ids.size(-1) - 1]], device=device)

                out_ng = chatglm_model(
                    input_ids         = input_for_inf,
                    past_key_values   = past,
                    # attention_mask    = attention_mk,
                    position_ids      = pos_for_inf,
                    use_cache         = True
                )
                logits_ng   = out_ng.logits[:, -1, :]
                raw_args.append(int(torch.argmax(logits_ng, dim=-1).item()))
                logits_ng   = logits_proc(inputs_ids, logits_ng)
                probs_ng    = torch.softmax(logits_ng, dim=-1)
                top5_token_ids = torch.topk(probs_ng, 5, dim=-1).indices.squeeze(0).tolist()
                next_token_id = torch.argmax(logits_ng, dim=-1, keepdim=True)  # shape [1,1]
                past_ng       = out_ng.past_key_values

            tid = next_token_id.item()
            token_counts[tid] += 1
            if not chatglm_tokenizer.decode([tid]).strip():
                whitespace_total += 1
                if whitespace_total > 3:
                    repetition_pen_loss = repetition_pen_loss + torch.tensor(
                        50.0 * float(whitespace_total - 3), dtype=torch.float32, device=device)

            if any(key_id in top5_token_ids for key_id in key_token_ids):
                key_probs = probs_ng[:, key_token_ids].sum()
                key_loss  = key_loss +  2*key_probs
            past          = past_ng                  
            inputs_ids    = torch.cat([inputs_ids, next_token_id], dim=-1)
            attention_mk  = torch.cat([attention_mk, torch.ones(1, 1, device=device)], dim=-1)
            if tid == chatglm_tokenizer.eos_token_id:
                if step == 0:            
                    eos_penalty = 1e5
                break
        generated_text = chatglm_tokenizer.decode(inputs_ids[0], skip_special_tokens=True)
        print(generated_text)
        input_ce = inputs_ids[:, :-1]
        outputs  = chatglm_model(input_ce, labels=input_ce)
        ppl_weight = float(os.environ.get("PPL_W", "1.0"))
        loss_ce  = ppl_weight*torch.exp(outputs.loss)
        key_penalty_weight = 5.0
        min_loss = key_penalty_weight * ((key_loss+1) ** 2 - 1)
        entropy_loss = entropy_weight_term(outputs.logits, len_q)
        rep_w = float(os.environ.get("REP_W", "30.0"))
        div_w = float(os.environ.get("DIV_W", "30.0"))
        if len(raw_args) >= 2:
            bigrams = list(zip(raw_args, raw_args[1:]))
            rep_bigram = 1.0 - len(set(bigrams)) / len(bigrams)
            distinct   = len(set(raw_args)) / len(raw_args)
        else:
            rep_bigram = 0.0; distinct = 1.0
        degeneration_penalty = torch.tensor(
            rep_w * rep_bigram + div_w * (1.0 - distinct), device=device)
        total_loss = (
            min_loss
            + loss_ce
            + repetition_pen_loss
            + entropy_loss
            + degeneration_penalty
            # + eos_penalty
        )
    print("loss:",min_loss.item(), "perplxity:",loss_ce.item(), "rep_pen:",repetition_pen_loss.item(),
          "entropy:",entropy_loss.item(), "degen:",degeneration_penalty.item(), "total_loss:",total_loss.item())
    return total_loss


