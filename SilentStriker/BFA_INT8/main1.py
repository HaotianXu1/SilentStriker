import os
import logging
logging.basicConfig(filename='./save/layer.log', force=True, level=logging.INFO, format='%(asctime)s - %(message)s', encoding='utf-8')
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
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from attack.BFA import *
import loss_2 as loss_func
import loss_1 as loss_func_back
import sys
import gc
import spacy
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test"))
if module_path not in sys.path:
    sys.path.append(module_path)

# model_name="Qwen2.5-14B-Instruct"
# model_name='Mistral-7B-Instruct'
# model_name="DeepSeek"
# model_name="Llama-3.1-8B-Instruct"
model_name="Qwen3-8B"


parser = argparse.ArgumentParser(description='BFA on LLM for one question')
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save logs and checkpoints.')
parser.add_argument('--n_iter', type=int, default=10, help='Number of attack iterations.')
parser.add_argument('--k_top', type=int, default=10, help='Top k weights with largest gradients to check for bit-flipping.')
parser.add_argument('--manualSeed', type=int, default=42, help='Manual random seed.')
parser.add_argument('--model_name', type=str, default=model_name, help='Manual random seed.')
args = parser.parse_args()


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")
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

    
)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


nlp = spacy.load("en_core_web_sm")
exclude_pos = {"DET", "ADP", "AUX", "CONJ", "PRON", "PUNCT", "CCONJ", "SCONJ", "VERB", "PART", "ADV"}
exclude_chars = {}
fixed_chars = {}
def get_key_tokens(prompt, tokenizer, model):
    device = model.device
    inputs_ids=tokenizer(prompt, return_tensors="pt",padding=True,truncation=True).input_ids.to(device)
    with torch.no_grad():
        model_output = model.generate(inputs_ids,max_new_tokens=40,no_repeat_ngram_size=2,repetition_penalty = 1.2,do_sample=False)
        generated_text = tokenizer.decode(model_output[0], skip_special_tokens=True)
    print(f"Model1 outputs: {generated_text}")
    doc1 = nlp(generated_text)
    key_tokens1 = [token.text for token in doc1 if token.pos_ not in exclude_pos]
    doc2 = nlp(prompt)
    key_tokens2 = [token.text for token in doc2 if token.pos_ not in exclude_pos]
    key_tokens=[x for x in key_tokens1 if x not in key_tokens2]
    key_tokens=list(set(key_tokens))
    key_tokens = [token for token in key_tokens if token not in exclude_chars]
    key_tokens_ = [" " + token for token in key_tokens]
    key_tokens=key_tokens+key_tokens_
    key_tokens_upper = [token.upper() for token in key_tokens]
    key_tokens += key_tokens_upper
    key_tokens.extend(fixed_chars)
    print(f"Key Tokens: {key_tokens}")
    key_token_ids = []
    for word in key_tokens:
        tokens = tokenizer.tokenize(word)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        filtered_ids = [token_id for token_id in ids if token_id != 220]
        key_token_ids.extend(filtered_ids)
        # if ids != 220:
        #     key_token_ids.extend(ids)
    key_token_ids=list(set(key_token_ids))
    print(f"Key token IDs: {key_token_ids}")
    return key_token_ids

def perform_attack(attacker, model,model_name, model_quan,clean_model, dataset, tokenizer, tokenizer_quan, N_iter,N_iter_recover, log, writer,forbidden_chars, flag,penalty_factor=50.0):
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
        key_token_ids = get_key_tokens(question, tokenizer, model_quan)
        key_token_ids_quan = key_token_ids
        # = get_key_tokens(question_clean, tokenizer, model_quan)
        key_token_ids_list.append(key_token_ids)
        key_token_ids_quan_list.append(key_token_ids_quan)

        loss = loss_func.loss_func(inputs_ids_quan, key_token_ids, model_quan, tokenizer_quan, penalty_factor=50.0)

        total_loss += loss
        print_log(f'Initial Loss for question "{question_clean}": {loss.item()}', log)
    
    print_log(f'Total Initial Loss: {total_loss.item()}', log)

    save_time=0
    for i_iter in range(N_iter):
        print_log(f'******** Iteration {i_iter+1} ********', log)
        moduel_name,flip_num=attacker.progressive_bit_search(model,model_name, model_quan, dataset, tokenizer, tokenizer_quan, model_quan.device,clean_model,i_iter+1,forbidden_chars, key_token_ids_list,key_token_ids_quan_list,flag,penalty_factor=50.0)
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
                model_output = model_quan.generate(inputs_ids_quan,max_new_tokens=100,no_repeat_ngram_size=2,repetition_penalty = 1.2,do_sample=False)
                output_answer = tokenizer_quan.decode(model_output[0], skip_special_tokens=True)
            print_log(f'after attack (Iteration {i_iter+1}): {output_answer}', log)
            loss_after_attack = loss_func.loss_func(inputs_ids_quan, key_token_ids_quan, model_quan, tokenizer_quan,clean_model,forbidden_chars, penalty_factor=50.0)
            total_loss_after_attack += loss_after_attack
        print_log(f'Total Loss after attack (Iteration {i_iter+1}): {total_loss_after_attack.item()}', log)
        writer.add_scalar('attack/total_loss_after_attack', total_loss_after_attack.item(), i_iter + 1)
        if i_iter + 1 in [10]:
            save_path = f"../../hugging_cache/{model_name}_8bit_afterattack_{i_iter+1}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f'Model saved after iteration {i_iter+1}')

        del inputs_ids_quan, key_token_ids_quan, model_output, output_answer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def print_log(print_string, log):
    print(print_string)
    log.write(print_string + '\n')
    log.flush()
    
# 主函数
def main():
    log = open(os.path.join(args.save_path, f'attack_log_{model_name}.txt'), 'w')
    tb_path = os.path.join(args.save_path, 'tb_log', 'run_' + str(args.manualSeed))
    writer = SummaryWriter(tb_path)
    model_path=f"../../hugging_cache/{model_name}"
    model, tokenizer = load_model(model_path)
    clean_model=0
    model_quan, tokenizer_quan = load_quan_model(model_path)
    attacker = BFA(criterion_back=loss_func_back.loss_func, criterion=loss_func.loss_func,name=model_name, k_top=args.k_top)
    with open("attack_dataset.txt", "r", encoding="utf-8") as file:
        dataset = eval(file.read())
    forbidden_chars_=["|","_","-","="," |"," -"," ="," _"]
    forbidden_chars=tokenizer(forbidden_chars_, return_tensors="pt").input_ids
    print(forbidden_chars)
    penalty_factor=50
    flag=0
    perform_attack(attacker, model,model_name, model_quan,clean_model, dataset, tokenizer, tokenizer_quan, args.n_iter,args.n_iter_recover, log, writer,forbidden_chars, flag,penalty_factor)
    log.close()

if __name__ == '__main__':
    main()