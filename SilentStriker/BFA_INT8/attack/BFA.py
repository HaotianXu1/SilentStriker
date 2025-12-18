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


    def float16_to_bin(self,value):
        packed = struct.pack('>e', value) 
        int_value = int.from_bytes(packed, "big")
        return f"{int_value:016b}"
    def bin_to_float16(self,binary_string):
        if len(binary_string) != 16:
            raise ValueError("Input must be a 16-bit binary string.")
        int_value = int(binary_string, 2)
        packed = int_value.to_bytes(2, byteorder="big")
    
        return struct.unpack('>e', packed)[0]
    def int8_to_bin(self, value):

        if value < 0:
            value = (1 << 8) + value  
        binary_str = f"{value:08b}" 
        return binary_str

    def bin_to_int8(self, binary_str):
        value = int(binary_str, 2)
        if value >= (1 << 7):  
            value -= (1 << 8) 
        return value
    def int8_fp16(self, int_param, scale,flag):
        fp16_param = (int_param * scale / 127).clone().detach().to(dtype=torch.float16)
        if torch.isnan(fp16_param).any():
            print("Warning: fp16_param contains NaN values")
        return fp16_param
    def col_ampere_to_row_major_flat_idx(self, flat_idx, rows, cols):
        TILE_SIZE = 32
        SUBTILE_SIZE = 8

        # Calculate base tile indices in col_ampere format
        global_tile_row = flat_idx // (TILE_SIZE ** 2)
        local_offset = flat_idx % (TILE_SIZE ** 2)

        # Decode local_row from local_offset
        local_row = local_offset // TILE_SIZE
        local_col = local_offset % TILE_SIZE

        # Reconstruct subrow in the original layout
        subrow = ((local_row // SUBTILE_SIZE) * 2 + (local_row % 2)) * SUBTILE_SIZE + (local_row % SUBTILE_SIZE) // 2

        # Reconstruct base_row and base_col in row-major format
        base_row = global_tile_row * TILE_SIZE
        base_col = (flat_idx % (rows * TILE_SIZE)) // TILE_SIZE * TILE_SIZE

        # Compute final row and column indices
        row_idx = base_row + subrow
        col_idx = base_col + local_col

        # Convert back to row-major flat index
        row_major_flat_idx = row_idx * cols + col_idx

        return row_major_flat_idx
    
    def row_major_to_col_ampere_flat_idx(self, flat_idx, rows, cols):
        TILE_SIZE = 32
        SUBTILE_SIZE = 8

        # Convert flat index to row and column indices in row-major format
        row_idx = flat_idx // cols
        col_idx = flat_idx % cols

        # Calculate base tile offsets in row-major format
        base_row = (row_idx // TILE_SIZE) * TILE_SIZE
        base_col = (col_idx // TILE_SIZE) * TILE_SIZE
        subrow = row_idx % TILE_SIZE

        # Calculate local_row in the col_ampere layout
        local_row = ((subrow % SUBTILE_SIZE) // 2) * SUBTILE_SIZE + (subrow // SUBTILE_SIZE) * 2 + (subrow % 2)

        # Calculate global_offset and row_offset
        global_offset = (base_col // TILE_SIZE) * rows * TILE_SIZE + (base_row // TILE_SIZE) * TILE_SIZE ** 2
        row_offset = local_row * TILE_SIZE

        # Final 1D offset in col_ampere
        col_ampere_flat_idx = global_offset + row_offset + (col_idx % TILE_SIZE)

        return col_ampere_flat_idx



    def flip_bit(self, m, m_quan,device, device_m, inputs_ids, labels_list, model_quan, tokenizer):
        m.N_bits = 8 
        row_n=m.weight.shape[0]
        col_n=m.weight.shape[1]
        k_top=self.k_top*10
        w_grad_topk_abs, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(k_top)
        w_idx_topk_ampere = [self.row_major_to_col_ampere_flat_idx(flat_idx.item(), row_n, col_n) for flat_idx in w_idx_topk]
        w_idx_topk_ampere = torch.tensor(w_idx_topk_ampere)
        weight_topk = m_quan.state.CxB.detach().view(-1)[w_idx_topk_ampere]
        device = w_idx_topk_ampere.device
        w_idx_topk_=w_idx_topk.clone().to(device)
        weight_topk_SCB = m_quan.state.SCB.view(-1,1).expand(-1,col_n).detach().reshape(-1)[w_idx_topk_]
        weight_topk_bin = []  
        weight = m_quan.state.CxB.detach().view(-1).clone()
        for param in weight_topk:
            weight_topk_bin.append(self.int8_to_bin(param.item())) 
        bit_pos_list = [7] 
        flipped_bin_values = []
        flipped_weight_topk = []
        flipped_weight_topk_origin = []
        flip_num=0
        for i, bin_value in enumerate(weight_topk_bin):
            grad_sign = torch.sign(m.weight.grad.detach().view(-1)[w_idx_topk[i]]).item()
            param_sign = torch.sign(weight_topk[i]).item()
            if (grad_sign < 0 and param_sign < 0) or (grad_sign > 0 and param_sign > 0):
                if flip_num<self.k_top:
                    flip_num+=1
                    flag=1
                    bin_value = bin_value[::-1]
                    flipped_bin = list(bin_value)
                    for bit_pos in bit_pos_list:
                        flipped_bin[bit_pos] = '1' if flipped_bin[bit_pos] == '0' else '0'
                    flipped_bin = ''.join(flipped_bin[::-1])
                    flipped_weight = self.bin_to_int8(flipped_bin)
                    flipped_weight = max(-128, min(127, flipped_weight))
                    flipped_bin_values.append(flipped_bin)
                    flipped_weight_topk.append(flipped_weight)
                    flipped_weight_topk_origin.append(self.int8_fp16(flipped_weight, weight_topk_SCB[i],flag))
                else:
                    flag=0
                    flipped_weight_topk.append(weight_topk[i].item())
                    flipped_weight_topk_origin.append(self.int8_fp16(weight_topk[i].item(), weight_topk_SCB[i],flag))
            else:
                flag=0
                flipped_weight_topk.append(weight_topk[i].item())
                flipped_weight_topk_origin.append(self.int8_fp16(weight_topk[i].item(), weight_topk_SCB[i],flag))
        print(flip_num)
        weight = m_quan.state.CxB.detach().view(-1).clone()
        weight[w_idx_topk_ampere] = torch.tensor(flipped_weight_topk, dtype=m_quan.state.CxB.dtype).to(weight.device)
        param_flipped = weight.view(m_quan.state.CxB.size())
        weight_origin = m.weight.detach().view(-1).clone()
        weight_origin[w_idx_topk] = torch.tensor(flipped_weight_topk_origin, dtype=m.weight.dtype, device=device_m)
        param_flipped_origin = weight_origin.view(m.weight.data.size())
        
        return param_flipped, param_flipped_origin,flip_num
    
    def progressive_bit_search(self, model,model_name, model_quan, dataset, tokenizer, tokenizer_quan, device,iter,forbidden_chars,key_token_ids_list,key_token_ids_quan_list, flag,penalty_factor=50.0):
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

        i=0
        for data in dataset:
            init_prompt="Please give the answer directly."
            question = data["question"]
            self.logger.info(f"Processing question: {question}")
            inputs_ids = tokenizer(question, return_tensors="pt",padding=True,truncation=True).input_ids.to(model.device)
            inputs_ids_quan = tokenizer_quan(question, return_tensors="pt",padding=True,truncation=True).input_ids.to(model_quan.device)
            inputs_ids_list.append(inputs_ids)
            inputs_ids_quan_list.append(inputs_ids_quan)
            key_token_ids=key_token_ids_list[i]
            key_token_ids_quan=key_token_ids_quan_list[i]
            total_loss += self.criterion_back(inputs_ids, key_token_ids, model, tokenizer,forbidden_chars, penalty_factor=50.0)
            total_loss_quan += self.criterion(inputs_ids_quan, key_token_ids_quan, model_quan, tokenizer_quan,forbidden_chars, penalty_factor=50.0)
            i+=1
        self.loss = total_loss
        self.loss_quan = total_loss_quan
        self.loss_quan_min = self.loss_quan.item()
        self.logger.info(f"Initial total_loss_quan_min: {self.loss_quan_min}")

        for m in model.modules():
            if hasattr(m, 'weight') and m.weight.grad is not None:
                m.weight.grad.data.zero_()
        self.loss.backward()
        self.logger.info(f"Starting bit flipping iteration, bits flipped so far: {self.n_bits2flip}")
        number=0
        
        for (name, module), (name2, module_quan) in zip(model.named_modules(), model_quan.named_modules()):
            if (name != 'lm_head' and name != "model.norm" and name != "model.embed_tokens" and
                        hasattr(module, 'weight') and "input_layernorm" not in name and "post_attention_layernorm" not in name and "norm" not in name):
                if number>=28 and number <=240:
                    number+=1
                    continue
                else:
                    print(name)
                    number+=1
                    clean_weight = module_quan.state.CxB.detach().clone()
                    clean_weight_origin = module.weight.data.detach().clone()
                    device = next(module_quan.parameters()).device
                    device_m = next(module.parameters()).device
                    with torch.no_grad():
                        attack_weight, attack_weight_origin,flip_num = self.flip_bit(
                            module, module_quan, device,device_m, inputs_ids_quan_list, labels_quan_list, model_quan, tokenizer_quan
                        )
                    # print(torch.sum(attack_weight-clean_weight))
                    module_quan.state.CxB = attack_weight
                    module.weight.data = attack_weight_origin
                    self.loss_dict[name] = sum(
                        self.criterion(inputs_ids_quan, key_token_ids_quan, model_quan, tokenizer_quan,forbidden_chars, penalty_factor=50.0)
                        for inputs_ids_quan, key_token_ids_quan in zip(inputs_ids_quan_list, key_token_ids_quan_list)
                    )
                    # if self.loss_dict[name]
                    self.logger.info(f"Module: {name}, Flip number: {flip_num}, Loss after bit flip: {self.loss_dict[name].item()}")
                    #  output: {out}
                    print(f"Module: {name}, Loss after bit flip: {self.loss_dict[name].item()}")
                    module_quan.state.CxB = clean_weight
                    module.weight.data = clean_weight_origin

        filtered_loss_dict = {k: v for k, v in self.loss_dict.items() if not math.isnan(v)}
        if filtered_loss_dict:
            min_loss_module = min(filtered_loss_dict.items(), key=operator.itemgetter(1))[0]
            # if self.loss_quan_max>self.loss_dict[max_loss_module]:
            #     self.k_top+=5
            self.loss_quan_min = self.loss_dict[min_loss_module]
            self.logger.info(f"Min loss module: {min_loss_module}, Final loss_min: {self.loss_quan_min}")

        for (name, module), (name2, module_quan) in zip(model.named_modules(), model_quan.named_modules()):
            if name == min_loss_module:
                device_m = next(module.parameters()).device
                attack_weight, attack_weight_origin,flip_num = self.flip_bit(
                    module, module_quan, device,device_m, inputs_ids_quan_list, labels_quan_list, model_quan, tokenizer_quan
                )
                module_quan.state.CxB = attack_weight
                module.weight.data = attack_weight_origin

        self.logger.info("Progressive bit search completed.")
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0
        model.zero_grad()
        torch.cuda.empty_cache()
        return min_loss_module,flip_num

