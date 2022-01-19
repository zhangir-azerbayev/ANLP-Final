import sys 
import random 
import math 
import torch 
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import TrainingArguments, Trainer 
from transformers.integrations import TensorBoardCallback
from transformers.trainer_pt_utils import get_parameter_names


from dataset import read_mathqapython, MathQAPython

from curriculum import CurriculumSampler

from tqdm import tqdm 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.mkdir("./train_results/curriculum_tb2")

batch_size = 32
weight_decay=0.01
lr = 6.0e-5

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

# parameter stuff 
decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]

# curriculum learning stuff 



class CurriculumTrainer(Trainer): 
    def get_train_dataloader(self) -> DataLoader: 
        sampler = CurriculumSampler(30000, 
                                    2500, 
                                    1.4, 
                                    0.1, 
                                    batch_size, 
                                    len(data)
                                    )

        return DataLoader(
                self.train_dataset, 
                collate_fn = self.data_collator, 
                num_workers = self.args.dataloader_num_workers, 
                pin_memory = self.args.dataloader_pin_memory, 
                batch_sampler = sampler
                )

# Optimizer and scheduler 
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

gamma = 0.95
steps_per_epoch = math.ceil(len(train_set)/batch_size)
lr_lambda = lambda step: gamma ** (step//steps_per_epoch)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# Training loop
training_args = TrainingArguments(output_dir="./train_results/curriculum_tb2",
                                  num_train_epochs=1,
                                  logging_steps = 500, 
                                  save_steps = 2000,
                                  weight_decay=weight_decay,
                                  warmup_steps = 100,
                                  logging_dir="./train_results/curriculum_tb2/log", 
                                  per_device_train_batch_size=batch_size
                                  )

def data_collator(data):
    return {'input_ids': torch.stack([f[1] for f in data]),
            'attention_mask': torch.stack([f[2] for f in data]),
            'labels': torch.stack([f[1] for f in data])
           }

tb_writer = SummaryWriter(log_dir="./train_results/curriculum_tb2/tb_log")
tb_callback = TensorBoardCallback(tb_writer)

CurriculumTrainer(model=model, args=training_args, train_dataset=train_set,
        data_collator=data_collator, optimizers=[optimizer, scheduler], callbacks=[tb_callback]).train()


