import torch 
import json 
from pathlib import Path 

def read_mathqapython(path): 
    path = Path(path)
    with open(path, 'rb') as f: 
        mathqapython_list = json.load(f)


    return mathqapython_list

class MathQAPython(torch.utils.data.Dataset): 
    def __init__(self, sorted_instances, tokenizer, max_length): 
        self.data = sorted_instances
        self.tokenizer = tokenizer 
        self.max_length = max_length
    

    def __getitem__(self, idx): 
        instance = self.data[idx]
        text = instance['text'] 
        solution = instance['text'] + '\n' + instance['code'] 
        answer = instance['answer']

        text_encode = self.tokenizer(text, 
                max_length=self.max_length, truncation=True, 
                padding='max_length', return_tensors='pt')
        solution_encode = self.tokenizer(solution, 
                max_length=self.max_length, truncation=True, 
                padding='max_length', return_tensors='pt')
        text_ids = text_encode['input_ids'].squeeze()
        solution_ids = solution_encode['input_ids'].squeeze()
        solution_attn = solution_encode['attention_mask'].squeeze()

        return text_ids.long(), solution_ids.long(), solution_attn.long(), answer


    def __len__(self): 
        return len(self.data) 

