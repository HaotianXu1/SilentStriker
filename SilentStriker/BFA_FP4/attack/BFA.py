import os
from pyexpat import model
import torch
import operator
from attack.data_conversion import *
import torch.nn as nn
import math
import logging
import numpy as np
import struct
import random
import spacy

class BFA(object):
    def __init__(self, criterion_back,criterion, name,k_top=2):
        self.criterion_back = criterion_back
        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        self.name= name
        self.logger = self.setup_logger()
        

    def setup_logger(self):
        logger = logging.getLogger("BFA")
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f"bfa_{self.name}.log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def print_log(self, print_string):
        self.logger.info(print_string)
   
    def int4_to_bin(self, value):
        if value < 0:
            value = (1 << 4) + value  
        binary_str = f"{value:04b}"  
        return binary_str

    def bin_to_int4(self, binary_str):
        value = int(binary_str, 2)
        if value >= (1 << 3): 
            value -= (1 << 4)  
        return value
    def int8_fp16(self, int_param, scale):
        target_values = torch.tensor([ 
         0.0000, -0.0052, -0.6667, -1.0000, -0.3333, -0.5000, -0.1667, -0.2500,0.0000,  0.0052,  0.6667,  1.0000,  0.3333,  0.5000,  0.1667,  0.2500])
        indices = int_param + 8
        indices = torch.tensor(indices, device=target_values.device)
        int_param = target_values[indices]
        scale = torch.tensor(scale, device=target_values.device)
        fp16_param = (int_param * scale ).clone().detach().to(dtype=torch.float16)
        if torch.isnan(fp16_param).any():
            print("Warning: fp16_param contains NaN values")
        return fp16_param


    def flip_bit(self, m, m_quan, device1,device_m, inputs_ids, labels_list, model_quan, tokenizer):
        '''
        Flip bits in the most significant bits of the top-k gradient elements.
        '''
        m.N_bits = 4 
        row=m.weight.shape[0]
        col=m.weight.shape[1]
        k_top=self.k_top*8
        w_grad_topk_abs, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(k_top)
        w_idx_topk_=w_idx_topk.clone().to(device1)
        abs_shape=int(row*col/64)
        abs_max=m_quan.quant_state.absmax
        abs_max=abs_max.view(-1,1).expand(-1,64).to(device_m)
        abs_max=abs_max.reshape(-1)
        abs_max_=abs_max[w_idx_topk]
        high_bits = (m_quan.weight >> 4) & 0x0F 
        low_bits = m_quan.weight & 0x0F          

        high_bits = high_bits.flatten()
        low_bits = low_bits.flatten()

        int4_weights = torch.empty(high_bits.size(0) * 2, dtype=torch.uint8).to(device1)
        int4_weights[0::2] = high_bits 
        int4_weights[1::2] = low_bits  
        int4_weights = int4_weights.view(row, col)
        int4_weights = int4_weights - (int4_weights >= 8) * 16
        

        weight_topk = int4_weights.detach().view(-1)[w_idx_topk_]
        weight_topk_bin = [] 
        weight = m_quan.weight.detach().view(-1).clone()
        int4_weights = int4_weights.flatten()
        for param in weight_topk:
            weight_topk_bin.append(self.int4_to_bin(param)) 
        flipped_bin_values = []
        flipped_weight_topk = []
        flipped_weight_topk_origin = []
        flip_num=0
        for i, bin_value in enumerate(weight_topk_bin):
            grad_sign = torch.sign(m.weight.grad.detach().view(-1)[w_idx_topk[i]]).item()
            param_sign = torch.sign(weight_topk[i]).item()
            if (((bin_value == '0000' or bin_value == '0001' or bin_value == '1110' or bin_value == '1111' ) and grad_sign<0)
                or ((bin_value == '1000' or bin_value == '1001' or bin_value == '0110' or bin_value == '0111') and grad_sign>0)
                or ((bin_value == '0010' or bin_value == '0011' or bin_value == '0100' or bin_value == '0101') and grad_sign>0)
                or ((bin_value == '1010' or bin_value == '1011' or bin_value == '1100' or bin_value == '1101') and grad_sign<0)
                ):

                if flip_num<self.k_top:
                    flip_num+=1
                    flipped_bin = list(bin_value)
                    
                    if (
                    bin_value == '0001' or   
                    bin_value == '1001'):
                        bit_pos_list=[2]
                    elif (bin_value == '1000' or bin_value == '0000'):
                        bit_pos_list=[2,3]
                    elif (bin_value == '0100'):
                        bit_pos_list=[0,1,2,3]
                    elif (bin_value == '0010' or 
                    bin_value == '0011' or  
                    bin_value == '0101' or
                    bin_value == '1010' or
                    bin_value == '1011' or  
                    bin_value == '1100' or 
                    bin_value == '1101' ):
                        bit_pos_list=[0]
                    elif (bin_value == '0110' or  
                    bin_value == '1110'):
                        bit_pos_list=[0,1,3]
                    elif (
                    bin_value == '0111' or  
                    bin_value == '1111'):
                        bit_pos_list=[0,1]

                    for bit_pos in bit_pos_list:
                        flipped_bin[bit_pos] = '1' if flipped_bin[bit_pos] == '0' else '0'
                    flipped_bin = ''.join(flipped_bin)
                    flipped_weight = self.bin_to_int4(flipped_bin)
                    flipped_weight = max(-8, min(7, flipped_weight))
                    flipped_bin_values.append(flipped_bin)
                    flipped_weight_topk.append(flipped_weight)
                    flipped_weight_topk_origin.append(self.int8_fp16(flipped_weight, abs_max_[i]))
                else:
                    flipped_bin_values.append(bin_value)
                    flipped_weight_topk.append(weight_topk[i].item())
                    flipped_weight_topk_origin.append(self.int8_fp16(weight_topk[i].item(), abs_max_[i]))
            else:
                flipped_bin_values.append(bin_value)
                flipped_weight_topk.append(weight_topk[i].item())
                flipped_weight_topk_origin.append(self.int8_fp16(weight_topk[i].item(), abs_max_[i]))

        int4_weights[w_idx_topk_] = torch.tensor(flipped_weight_topk,  device=device1)
        high_bits = int4_weights[0::2]  
        high_bits=high_bits+(high_bits<0)*16
        low_bits = int4_weights[1::2]  
        low_bits=low_bits+(low_bits<0)*16
        uint8_weights = (high_bits << 4) | low_bits
        param_flipped = uint8_weights.to(dtype=torch.uint8).view(m_quan.weight.data.size())
        weight_origin = m.weight.detach().view(-1).clone()
        weight_origin[w_idx_topk] = torch.tensor(flipped_weight_topk_origin, dtype=m.weight.dtype, device=device_m)
        param_flipped_origin = weight_origin.view(m.weight.data.size())
        
        return param_flipped, param_flipped_origin,flip_num

    
    def progressive_bit_search(self, model,model_name, model_quan, dataset, tokenizer, tokenizer_quan, device,alpha,beta,temp,kappa,clean_model,iter,forbidden_chars,key_token_ids_list,key_token_ids_quan_list, flag,penalty_factor=50.0):
        self.logger.info("Starting progressive bit search...")
        self.loss_dict.clear()
        model.eval()
        model_quan.eval()

        total_loss = 0
        total_loss_quan = 0
        inputs_ids_list = []
        inputs_ids_quan_list = []
        labels_list = []
        labels_quan_list = []
        key_token_ids_list=key_token_ids_list
        key_token_ids_quan_list=key_token_ids_quan_list
        # 遍历数据集并记录日志
        i=0
        for data in dataset:
            init_prompt=""
            question = data["question"]+init_prompt
            self.logger.info(f"Processing question: {question}")
            inputs_ids = tokenizer(question, return_tensors="pt",padding=True,truncation=True).input_ids.to(model.device)
            inputs_ids_quan = tokenizer_quan(question, return_tensors="pt",padding=True,truncation=True).input_ids.to(model_quan.device)
            inputs_ids_list.append(inputs_ids)
            inputs_ids_quan_list.append(inputs_ids_quan)
            key_token_ids=key_token_ids_list[i]
            key_token_ids_quan=key_token_ids_quan_list[i]
            total_loss += self.criterion_back(inputs_ids, key_token_ids, model, tokenizer,alpha,beta,temp,kappa,clean_model,forbidden_chars, penalty_factor=50.0)
            total_loss_quan += self.criterion(inputs_ids_quan, key_token_ids_quan, model_quan, tokenizer_quan,alpha,beta,temp,kappa,clean_model,forbidden_chars, penalty_factor=50.0)
            i+=1
        self.loss = total_loss
        self.loss_quan = total_loss_quan
        self.loss_quan_min = self.loss_quan.item()
        self.logger.info(f"Initial total_loss_quan_min: {self.loss_quan_min}")
        for m in model.modules():
            if hasattr(m, 'weight') and m.weight.grad is not None:
                m.weight.grad.data.zero_()
        self.loss.backward()
        plus_flag=0
        while self.loss_quan_min >= self.loss_quan.item():
            self.n_bits2flip += 1
            if self.n_bits2flip >=2:
                self.k_top+=20
                plus_flag=1
            self.logger.info(f"Starting bit flipping iteration, bits flipped so far: {self.n_bits2flip}")
            number=0
            
            for (name, module), (name2, module_quan) in zip(model.named_modules(), model_quan.named_modules()):
                if (name != 'lm_head' and name != "model.norm" and name != "model.embed_tokens" and
                        hasattr(module, 'weight') and "input_layernorm" not in name and "post_attention_layernorm" not in name and "layernorm" not in name and "norm" not in name):
                    if number>=28 and number <=200:
                        number+=1
                        continue
                    print(name)
                    number+=1
                    clean_weight = module_quan.weight.data.detach().clone()
                    clean_weight_origin = module.weight.data.detach().clone()
                    device = next(module_quan.parameters()).device
                    device_m = next(module.parameters()).device
                    with torch.no_grad():
                        attack_weight, attack_weight_origin,flip_num = self.flip_bit(
                            module, module_quan, device,device_m, inputs_ids_quan_list, labels_quan_list, model_quan, tokenizer_quan
                        )
                    module_quan.weight.data = attack_weight
                    module.weight.data = attack_weight_origin
                    
                    self.loss_dict[name] = sum(
                        self.criterion(inputs_ids_quan, key_token_ids_quan, model_quan, tokenizer_quan,alpha,beta,temp,kappa,clean_model,forbidden_chars, penalty_factor=50.0)
                        for inputs_ids_quan, key_token_ids_quan in zip(inputs_ids_quan_list, key_token_ids_quan_list)
                    )
                    self.logger.info(f"Module: {name}, Flip number: {flip_num}, Loss after bit flip: {self.loss_dict[name].item()}")
                    print(f"Module: {name}, Loss after bit flip: {self.loss_dict[name].item()}")
                    folder_name = f"./attack_processing/iter_{iter}"
                    os.makedirs(folder_name, exist_ok=True)
                    module_quan.weight.data = clean_weight
                    module.weight.data = clean_weight_origin

            filtered_loss_dict = {k: v for k, v in self.loss_dict.items() if not math.isnan(v)}
            if filtered_loss_dict:
                min_loss_module = min(filtered_loss_dict.items(), key=operator.itemgetter(1))[0]
                self.loss_quan_min = self.loss_dict[min_loss_module]
                self.logger.info(f"Min loss module: {min_loss_module}, Final loss_min: {self.loss_quan_min}")
        for (name, module), (name2, module_quan) in zip(model.named_modules(), model_quan.named_modules()):
            if name == min_loss_module:
                device_m = next(module.parameters()).device
                attack_weight, attack_weight_origin,flip_num = self.flip_bit(
                    module, module_quan, device,device_m, inputs_ids_quan_list, labels_quan_list, model_quan, tokenizer_quan
                )
                module_quan.weight.data = attack_weight
                module.weight.data = attack_weight_origin
        if plus_flag==1:
            self.k_top+=-20
            plus_flag=0
        self.logger.info("Progressive bit search completed.")
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0
        model.zero_grad()
        torch.cuda.empty_cache()
        return min_loss_module,flip_num

