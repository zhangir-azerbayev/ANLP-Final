import sys 
import random 
import math 
import torch 
from torch.utils.data import DataLoader
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import TrainingArguments, Trainer 

from dataset import read_mathqapython, MathQAPython

from curriculum import CurriculumSampler

from tqdm import tqdm 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Tokenizer 
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token 
max_length = 450
# Loading data
print("loading data")
data = read_mathqapython('data/mathqapython_train.json')

def seqlen(instance): 
    return len(tokenizer.encode(instance['text']+instance['code']))

sorted_data = sorted(data, key=seqlen)
train_set = MathQAPython(sorted_data, tokenizer, max_length)

# model 
print('loading model')
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

# curriculum learning stuff 



class CurriculumTrainer(Trainer): 
    def get_train_dataloader(self) -> DataLoader: 
        sampler = CurriculumSampler(12000, 
                                    1000, 
                                    1.4, 
                                    0.1, 
                                    8, 
                                    len(data)
                                    )

        return DataLoader(
                self.train_dataset, 
                collate_fn = self.data_collator, 
                num_workers = self.args.dataloader_num_workers, 
                pin_memory = self.args.dataloader_pin_memory, 
                batch_sampler = sampler
                )



# Training loop
training_args = TrainingArguments(output_dir="./train_results/curriculum2",
                                  num_train_epochs=1,
                                  logging_steps = 500, 
                                  save_steps = 500,
                                  weight_decay=0.01,
                                  warmup_steps = 100,
                                  logging_dir="./train_results/curriculum2/log"
                                  )

def data_collator(data):
    return {'input_ids': torch.stack([f[1] for f in data]),
            'attention_mask': torch.stack([f[2] for f in data]),
            'labels': torch.stack([f[1] for f in data])
           }

CurriculumTrainer(model=model, args=training_args, train_dataset=train_set,
        data_collator=data_collator).train()


