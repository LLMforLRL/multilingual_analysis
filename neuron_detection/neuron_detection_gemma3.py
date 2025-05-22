import os
from dataclasses import field, dataclass
from typing import Optional, Any
import transformers
from rouge_score import rouge_scorer
import random
from itertools import groupby
import pdb
import re
import sys
from tqdm import tqdm
from typing import List
import logging
logging.basicConfig(level=logging.INFO)
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import csv

random.seed(112)


model_name = "google/gemma-3-12b-it"
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")



def Prompting(model, prompt, candidate_premature_layers):
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    hidden_states, outputs, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, activate_keys_o, layer_keys = model.generate(**{'input_ids':inputs.input_ids, 'max_new_tokens':1, 'candidate_premature_layers':candidate_premature_layers})
    hidden_embed = {}
    # pdb.set_trace()
    for i, early_exit_layer in enumerate(candidate_premature_layers):
        hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][0])
        # knowledge_neurons_word[early_exit_layer] = tokenizer.decode(knowledge_neurons[early_exit_layer][0])
        # hidden_info[early_exit_layer] = tokenizer.decode(torch.tensor(hidden_values[early_exit_layer]).to("cuda"))
    answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return hidden_embed, answer, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, activate_keys_o, layer_keys


# Patch both Gemma2DecoderLayer.forward definitions that appear in the file
# (they're duplicated for the Flash-Attention and SDPA versions) so the else
# branch also returns a seventh dummy element, e.g.：

# diff
# Copy
# Edit
# -    else:
# -        return attn_output, attn_weights, past_key_value_real, [], [], []
# +    else:
# +        # pad with an empty list for o-scores so the tuple length is always 7
# +        return attn_output, attn_weights, past_key_value_real, [], [], [], []

def main(argv):

    lines = []
    # file_path = "./corpus_all/"+argv[0] + ".txt"
    file_path = "./corpus_nllb/"+"train_100k."+argv[0]

    with open(file_path,'r') as file:
        lines = file.readlines()
    # print(len(lines))
    # lines = [line.strip() for line in lines]
    # print(int(argv[1]))
    # lines = random.sample(lines, int(argv[1]))
    lines = [line.strip() for line in lines]
    print(len(lines))
    k = int(argv[1])               # how many the user asked for
    if k > len(lines) or k <= 0:    # too big (or negative) → take all
        k = len(lines)

    lines = random.sample(lines, k)


    candidate_premature_layers = []
    for i in range(32):
        candidate_premature_layers.append(i)


    activate_keys_set_fwd_up = []
    activate_keys_set_fwd_down = []
    activate_keys_set_q = []
    activate_keys_set_k = []
    activate_keys_set_v = []

    count = 0

    for prompt in tqdm(lines):
        # print(prompt)
        model.eval()
        try:
            with torch.no_grad(): 
                hidden_embed, answer, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, _, _ = Prompting(model, prompt, candidate_premature_layers)
            
            activate_keys_set_fwd_up.append(activate_keys_fwd_up)
            activate_keys_set_fwd_down.append(activate_keys_fwd_down)
            activate_keys_set_q.append(activate_keys_q)
            activate_keys_set_k.append(activate_keys_k)
            activate_keys_set_v.append(activate_keys_v)
            torch.cuda.empty_cache()
        except Exception as e:
            count += 1
            # Handle the OutOfMemoryError here
            print(count)
            print(e)

    # Initialize dictionary for common elements
    common_elements_dict_fwd_up = {}
    common_elements_dict_fwd_down = {}
    common_elements_dict_q = {}
    common_elements_dict_k = {}
    common_elements_dict_v = {}


    # Iterate through the keys of the first dictionary
    for key in activate_keys_set_fwd_up[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_fwd_up):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_fwd_up]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_fwd_up[key] = common_elements

    for key in activate_keys_set_fwd_down[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_fwd_down):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_fwd_down]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_fwd_down[key] = common_elements
    # print(common_elements_dict_fwd_down)


    for key in activate_keys_set_q[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_q):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_q]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_q[key] = common_elements
    # print(common_elements_dict_q)


    for key in activate_keys_set_k[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_k):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_k]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_k[key] = common_elements

    for key in activate_keys_set_v[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_v):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_v]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_v[key] = common_elements
    # print(common_elements_dict_v)
    os.makedirs("./output_neurons_gemma3", exist_ok=True)


    file_path = "./output_neurons_gemma3/" + model_name + argv[0] + "gsm_2000_12000_"+str(int(argv[1])-count)+".txt"

    with open(file_path,'w') as file:
        file.write(str(common_elements_dict_fwd_up) + '\n')
        file.write(str(common_elements_dict_fwd_down) + '\n')
        file.write(str(common_elements_dict_q) + '\n')
        file.write(str(common_elements_dict_k) + '\n')
        file.write(str(common_elements_dict_v) + '\n')




if __name__ == "__main__":
    main(sys.argv[1:])