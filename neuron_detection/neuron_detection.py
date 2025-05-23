# import os
# from dataclasses import field, dataclass
# from typing import Optional, Any
# import transformers
# from rouge_score import rouge_scorer
# import random
# from itertools import groupby
# import pdb
# import re
# import sys
# from tqdm import tqdm
# from typing import List
# import logging
# logging.basicConfig(level=logging.INFO)
# import torch
# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_dataset
# import csv

# random.seed(112)


# model_name = "google/gemma-2-9b-it"
# # model_name = "mistralai/Mistral-7B-Instruct-v0.2"


# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")



# def Prompting(model, prompt, candidate_premature_layers):
    
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     hidden_states, outputs, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, activate_keys_o, layer_keys = model.generate(**{'input_ids':inputs.input_ids, 'max_new_tokens':1, 'candidate_premature_layers':candidate_premature_layers})
#     hidden_embed = {}
#     # pdb.set_trace()
#     for i, early_exit_layer in enumerate(candidate_premature_layers):
#         hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][0])
#         # knowledge_neurons_word[early_exit_layer] = tokenizer.decode(knowledge_neurons[early_exit_layer][0])
#         # hidden_info[early_exit_layer] = tokenizer.decode(torch.tensor(hidden_values[early_exit_layer]).to("cuda"))
#     answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
#     answer = answer.replace('</s>', '')
    
#     return hidden_embed, answer, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, activate_keys_o, layer_keys


# # Patch both Gemma2DecoderLayer.forward definitions that appear in the file
# # (they're duplicated for the Flash-Attention and SDPA versions) so the else
# # branch also returns a seventh dummy element, e.g.：

# # diff
# # Copy
# # Edit
# # -    else:
# # -        return attn_output, attn_weights, past_key_value_real, [], [], []
# # +    else:
# # +        # pad with an empty list for o-scores so the tuple length is always 7
# # +        return attn_output, attn_weights, past_key_value_real, [], [], [], []

# def main(argv):

#     lines = []
#     # file_path = "./corpus_all/"+argv[0] + ".txt"
#     file_path = "./corpus_nllb/"+"train_100k."+argv[0]

#     with open(file_path,'r') as file:
#         lines = file.readlines()
#     # print(len(lines))
#     # lines = [line.strip() for line in lines]
#     # print(int(argv[1]))
#     # lines = random.sample(lines, int(argv[1]))
#     lines = [line.strip() for line in lines]
#     print(len(lines))
#     k = int(argv[1])               # how many the user asked for
#     if k > len(lines) or k <= 0:    # too big (or negative) → take all
#         k = len(lines)

#     lines = random.sample(lines, k)


#     candidate_premature_layers = []
#     for i in range(32):
#         candidate_premature_layers.append(i)


#     activate_keys_set_fwd_up = []
#     activate_keys_set_fwd_down = []
#     activate_keys_set_q = []
#     activate_keys_set_k = []
#     activate_keys_set_v = []

#     count = 0

#     for prompt in tqdm(lines):
#         # print(prompt)
#         model.eval()
#         try:
#             with torch.no_grad(): 
#                 hidden_embed, answer, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, _, _ = Prompting(model, prompt, candidate_premature_layers)
            
#             activate_keys_set_fwd_up.append(activate_keys_fwd_up)
#             activate_keys_set_fwd_down.append(activate_keys_fwd_down)
#             activate_keys_set_q.append(activate_keys_q)
#             activate_keys_set_k.append(activate_keys_k)
#             activate_keys_set_v.append(activate_keys_v)
#             torch.cuda.empty_cache()
#         except Exception as e:
#             count += 1
#             # Handle the OutOfMemoryError here
#             print(count)
#             print(e)
#             continue

#     # Initialize dictionary for common elements
#     common_elements_dict_fwd_up = {}
#     common_elements_dict_fwd_down = {}
#     common_elements_dict_q = {}
#     common_elements_dict_k = {}
#     common_elements_dict_v = {}


#     # Iterate through the keys of the first dictionary
#     for key in activate_keys_set_fwd_up[0].keys():
#         # Check if the key exists in all dictionaries
#         if all(key in d for d in activate_keys_set_fwd_up):
#             # Extract corresponding arrays and find common elements
#             arrays = [d[key] for d in activate_keys_set_fwd_up]
#             common_elements = set.intersection(*map(set, arrays))

#             # Add common elements to the dictionary
#             common_elements_dict_fwd_up[key] = common_elements

#     for key in activate_keys_set_fwd_down[0].keys():
#         # Check if the key exists in all dictionaries
#         if all(key in d for d in activate_keys_set_fwd_down):
#             # Extract corresponding arrays and find common elements
#             arrays = [d[key] for d in activate_keys_set_fwd_down]
#             common_elements = set.intersection(*map(set, arrays))

#             # Add common elements to the dictionary
#             common_elements_dict_fwd_down[key] = common_elements
#     # print(common_elements_dict_fwd_down)


#     for key in activate_keys_set_q[0].keys():
#         # Check if the key exists in all dictionaries
#         if all(key in d for d in activate_keys_set_q):
#             # Extract corresponding arrays and find common elements
#             arrays = [d[key] for d in activate_keys_set_q]
#             common_elements = set.intersection(*map(set, arrays))

#             # Add common elements to the dictionary
#             common_elements_dict_q[key] = common_elements
#     # print(common_elements_dict_q)


#     for key in activate_keys_set_k[0].keys():
#         # Check if the key exists in all dictionaries
#         if all(key in d for d in activate_keys_set_k):
#             # Extract corresponding arrays and find common elements
#             arrays = [d[key] for d in activate_keys_set_k]
#             common_elements = set.intersection(*map(set, arrays))

#             # Add common elements to the dictionary
#             common_elements_dict_k[key] = common_elements

#     for key in activate_keys_set_v[0].keys():
#         # Check if the key exists in all dictionaries
#         if all(key in d for d in activate_keys_set_v):
#             # Extract corresponding arrays and find common elements
#             arrays = [d[key] for d in activate_keys_set_v]
#             common_elements = set.intersection(*map(set, arrays))

#             # Add common elements to the dictionary
#             common_elements_dict_v[key] = common_elements
#     # print(common_elements_dict_v)
#     os.makedirs("./output_neurons", exist_ok=True)


#     file_path = "./output_neurons/" + model_name + argv[0] + "gsm_2000_12000_"+str(int(argv[1])-count)+".txt"

#     with open(file_path,'w') as file:
#         file.write(str(common_elements_dict_fwd_up) + '\n')
#         file.write(str(common_elements_dict_fwd_down) + '\n')
#         file.write(str(common_elements_dict_q) + '\n')
#         file.write(str(common_elements_dict_k) + '\n')
#         file.write(str(common_elements_dict_v) + '\n')




# if __name__ == "__main__":
#     main(sys.argv[1:])
























import os
import gc
import logging
import random
import sys
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, List, Optional

import torch
import transformers
from rouge_score import rouge_scorer  # noqa: F401  # kept because the original script imports it
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------------
# Helper: recursively detach tensors and move them to CPU so they do not hold GPU RAM
# ----------------------------------------------------------------------------------

def _to_cpu(obj):
    """Detach *any* torch.Tensor inside *obj* and move it to CPU.

    Keeps non‑tensor leaves untouched so the structure (list/dict/str/...) is preserved.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, list):
        return [_to_cpu(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj  # primitives / strings


# ---------------------
# Original configuration
# ---------------------

random.seed(112)
model_name = "google/gemma-2-9b-it"  # "mistralai/Mistral-7B-Instruct-v0.2" also works

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 ,device_map="auto")
model.eval()

# ----------------------------------------------------------------------------------
# Core prompting function – *now* guarantees no GPU tensors survive the call
# ----------------------------------------------------------------------------------

def Prompting(model, prompt: str, candidate_premature_layers: List[int]):
    # Encode on CPU first
    inputs = tokenizer(prompt, return_tensors="pt")

    # ── NEW: follow the model’s primary device ─────────────────────────────
    first_param_dev = next(model.parameters()).device        # e.g. cuda:0 or cpu
    if first_param_dev.type == "cuda":
        inputs = inputs.to(first_param_dev)
    # ----------------------------------------------------------------------

    with torch.no_grad():
        (
            hidden_states,
            outputs,
            activate_keys_fwd_up,
            activate_keys_fwd_down,
            activate_keys_q,
            activate_keys_k,
            activate_keys_v,
            activate_keys_o,   # kept for compatibility
            layer_keys,
        ) = model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=1,
            use_cache=False,
            candidate_premature_layers=candidate_premature_layers,
        )


    # 3️⃣ Build the lightweight (string) representation we need
    hidden_embed = {
        early_layer: tokenizer.decode(hidden_states[early_layer][0])
        for early_layer in candidate_premature_layers
    }

    answer = tokenizer.decode(outputs[0]).replace("<pad> ", "").replace("</s>", "")

    # 4️⃣ *Immediately* detach everything that is still a tensor → CPU
    activate_keys_fwd_up = _to_cpu(activate_keys_fwd_up)
    activate_keys_fwd_down = _to_cpu(activate_keys_fwd_down)
    activate_keys_q = _to_cpu(activate_keys_q)
    activate_keys_k = _to_cpu(activate_keys_k)
    activate_keys_v = _to_cpu(activate_keys_v)

    # 5️⃣ Drop GPU‑resident objects & splash the allocator
    del inputs, hidden_states, outputs  # remaining GPU tensors
    torch.cuda.empty_cache()
    gc.collect()

    return (
        hidden_embed,
        answer,
        activate_keys_fwd_up,
        activate_keys_fwd_down,
        activate_keys_q,
        activate_keys_k,
        activate_keys_v,
    )


# ----------------------------------------------------------------------------------
# Main script – *unchanged structure*, only adds memory‑release hygiene
# ----------------------------------------------------------------------------------

def main(argv):
    # ------------------------
    # 1. Corpus initialisation
    # ------------------------
    file_path = "./corpus_nllb/" + "train_100k." + argv[0]
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    k = int(argv[1])
    if k > len(lines) or k <= 0:
        k = len(lines)
    lines = random.sample(lines, k)

    candidate_premature_layers = list(range(32))

    activate_keys_set_fwd_up: List[Any] = []
    activate_keys_set_fwd_down: List[Any] = []
    activate_keys_set_q: List[Any] = []
    activate_keys_set_k: List[Any] = []
    activate_keys_set_v: List[Any] = []

    skipped = 0

    # ------------------------------
    # 2. Iterate over sampled prompts
    # ------------------------------
    for prompt in tqdm(lines):
        try:
            (
                _hidden_embed,  # not used downstream, kept for completeness
                _answer,        # not used downstream
                act_fwd_up,
                act_fwd_down,
                act_q,
                act_k,
                act_v,
            ) = Prompting(model, prompt, candidate_premature_layers)

            # Lists now contain *CPU* objects only → safe to keep indefinitely
            activate_keys_set_fwd_up.append(act_fwd_up)
            activate_keys_set_fwd_down.append(act_fwd_down)
            activate_keys_set_q.append(act_q)
            activate_keys_set_k.append(act_k)
            activate_keys_set_v.append(act_v)

        except torch.cuda.OutOfMemoryError as e:
            skipped += 1
            logging.warning(f"Skipped a prompt due to OOM: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

    # ---------------------------------------------------
    # 3. Aggregate common elements across the prompt set
    # ---------------------------------------------------
    if not activate_keys_set_fwd_up:
        logging.error("No successful prompts – aborting aggregation.")
        return

    common_elements_dict_fwd_up = {}
    common_elements_dict_fwd_down = {}
    common_elements_dict_q = {}
    common_elements_dict_k = {}
    common_elements_dict_v = {}

    def _common(dest: dict, sets: List[dict]):
        for key in sets[0].keys():
            if all(key in d for d in sets):
                arrays = [d[key] for d in sets]
                dest[key] = set.intersection(*map(set, arrays))

    _common(common_elements_dict_fwd_up, activate_keys_set_fwd_up)
    _common(common_elements_dict_fwd_down, activate_keys_set_fwd_down)
    _common(common_elements_dict_q, activate_keys_set_q)
    _common(common_elements_dict_k, activate_keys_set_k)
    _common(common_elements_dict_v, activate_keys_set_v)

    # ----------------------------------
    # 4. Persist results exactly as before
    # ----------------------------------
    os.makedirs("./output_neurons", exist_ok=True)
    out_file = (
        f"./output_neurons/{model_name}{argv[0]}gsm_2000_12000_{len(lines)-skipped}.txt"
    )
    with open(out_file, "w") as f:
        f.write(str(common_elements_dict_fwd_up) + "\n")
        f.write(str(common_elements_dict_fwd_down) + "\n")
        f.write(str(common_elements_dict_q) + "\n")
        f.write(str(common_elements_dict_k) + "\n")
        f.write(str(common_elements_dict_v) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])