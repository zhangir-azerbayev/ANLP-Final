import sys
import re
import random
import yaml
import json
import os
import itertools
from tqdm import tqdm 
random.seed(1)

import numpy as np 
np.random.seed(1)

import torch
from torch.utils.data import DataLoader
torch.manual_seed(1)

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from dataset import read_mathqapython

from execution import semisafe_evaluate

device = 'cuda:1'

model_path = sys.argv[1]
outfile = sys.argv[2]

# Loads model and data 
print('loading model and data...')
model = GPTNeoForCausalLM.from_pretrained(model_path).to(device)

# Test set
data = read_mathqapython('data/mathqapython_test.json')
random.shuffle(data)
data = data[:1000]

# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token


def pass_k(lst, k): 
    """
    lst: Boolean list 
    k: value of pass@k to calculate. 
    """
    n = len(lst)
    c = sum(lst)
    if n - c < k: return 1.0 
    return 1.0 - np.prod(1.0 - k / 
                        np.arange(n-c+1, n+1))


results = []
for instance in tqdm(data): 
    label = instance['answer']
    encoded_prompt = tokenizer.encode(instance['text'], 
            return_tensors='pt').to(device)
    prompt_length = torch.numel(encoded_prompt)
    with torch.no_grad(): 
        out = model.generate(
                input_ids=encoded_prompt,
                do_sample=True ,
                temperature=0.2, 
                max_length = min([prompt_length+250, 450]), 
                pad_token_id=tokenizer.eos_token_id, 
                num_return_sequences=10
                )
        generated_ids = [ids[prompt_length:] for ids in out]

    untrunced_bodies = [tokenizer.decode(sample, skip_special_tokens=True)
            for sample in generated_ids]
    
    re_key = '^answer.*?\n'
    bodies = [completion[:re.search(re_key, completion).span()[1]]
            if re.search(re_key, completion) else completion 
            for completion in untrunced_bodies]

    answers = [semisafe_evaluate(program, 'answer', 1) for program in bodies]

    passed_lst = [(abs((answer - label)/label) < 0.01)
            if isinstance(answer, float) else False for answer in answers]

    result = dict()

    result[10] = pass_k(passed_lst, 10)
    result[1] = pass_k(passed_lst, 1)

    if True in passed_lst: 
        best_completion = bodies[passed_lst.index(True)]
    else: 
        best_completion = bodies[0]

    result['best_completion'] = best_completion 

    results.append(result)

pass10scores = [instance[10] for instance in results]
pass10average = sum(pass10scores)/len(pass10scores)
print("pass 10: ", pass10average)

pass1scores = [instance[1] for instance in results]
pass1average = sum(pass1scores)/len(pass1scores)
print("pass 1: ", pass1average)

to_dump = {'pass1': pass1average, 
        'pass10': pass10average, 
        'results': results}

with open(outfile, 'w') as fle: 
    json.dump(to_dump, fle)






