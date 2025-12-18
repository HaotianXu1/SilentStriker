import os
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
def loss_func(
        inputs_ids, key_token_ids,
        chatglm_model, chatglm_tokenizer,
        penalty_factor=10.0, max_length=50,
        repetition_penalty=1.2, ngram_size=2):

    device       = inputs_ids.device
    len_q        = inputs_ids.size(-1)         
    attention_mk = torch.ones(1, len_q, device=device)
    position_ids = torch.arange(0, len_q, device=device).unsqueeze(0)

    logits_proc  = LogitsProcessorList([
        NoRepeatNGramLogitsProcessor(ngram_size),
        RepetitionPenaltyLogitsProcessor(repetition_penalty),
    ])

    key_loss            = 0.0
    eos_penalty         = 0.0
    repetition_pen_loss = 0.0
    token_counts        = defaultdict(int)

    past = None
    for step in range(max_length):
        with torch.no_grad():
            if past is None:
                input_for_inf = inputs_ids
                pos_for_inf = position_ids
            else:
                input_for_inf = next_token_id
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
            logits_ng   = logits_proc(inputs_ids, logits_ng)
            probs_ng    = torch.softmax(logits_ng, dim=-1)
            top5_token_ids = torch.topk(probs_ng, 5, dim=-1).indices.squeeze(0).tolist()
            next_token_id = torch.argmax(logits_ng, dim=-1, keepdim=True)  
            past_ng       = out_ng.past_key_values

        tid = next_token_id.item()
        token_counts[tid] += 1
        if token_counts[tid] > 1:
            repetition_pen_loss += torch.exp(
                torch.tensor(token_counts[tid] - 3, dtype=torch.float32, device=device))

        if any(key_id in top5_token_ids for key_id in key_token_ids):
            out_grad = chatglm_model(
                input_ids       = input_for_inf,
                past_key_values = past,         
                # attention_mask  = attention_mk,
                position_ids    = pos_for_inf,
                use_cache       = True          
            )
            logits_g   = out_grad.logits[:, -1, :]
            logits_g   = logits_proc(inputs_ids, logits_g)
            probs_g    = torch.softmax(logits_g, dim=-1)
            key_probs  = probs_g[:, key_token_ids].sum()
            key_loss   = key_loss + key_probs        
        past          = past_ng                    
        inputs_ids    = torch.cat([inputs_ids, next_token_id], dim=-1)
        attention_mk  = torch.cat([attention_mk, torch.ones(1, 1, device=device)], dim=-1)

        if tid == chatglm_tokenizer.eos_token_id:
            if step == 0:           
                eos_penalty = 1e5
            break

    input_ce = inputs_ids[:, :-1]         
    outputs  = chatglm_model(input_ce, labels=input_ce)
    loss_ce  = torch.exp(outputs.loss)

    min_loss = (key_loss) ** 2         
    total_loss = (
        min_loss
        + loss_ce
        # + eos_penalty           
    )
    return total_loss

