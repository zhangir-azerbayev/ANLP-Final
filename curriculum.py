import math 
import random 
from torch.utils.data import Sampler 

def pacer(itr, n, step_length, increase, start_prop): 
    return round(n * min(1, start_prop * math.pow(increase, math.floor(itr / step_length))))

def get_instance(num_instances): 
    while True: 
        idxs = list(range(num_instances))
        random.shuffle(idxs)
        for idx in idxs: 
            yield idx



class CurriculumSampler(Sampler): 
    def __init__(self, num_iters, iters_per_refresh, increase, start_prop, 
            batch_size, n): 
        self.num_iters = num_iters
        self.iters_per_refresh = iters_per_refresh
        self.increase = increase
        self.start_prop = start_prop
        self.batch_size = batch_size 
        self.n = n 

    def __iter__(self): 
        num_iters = self.num_iters
        iters_per_refresh = self.iters_per_refresh
        increase = self.increase 
        start_prop = self.start_prop
        batch_size = self.batch_size
        n = self.n 

        for itr in range(num_iters): 
            if itr % iters_per_refresh == 0:
                # Do refresh 
                num_instances = pacer(itr, n, iters_per_refresh, increase, start_prop)
                instance_gen = get_instance(num_instances)
            batch = [next(instance_gen) for _ in range(batch_size)]
            yield batch 

    def __len__(self): 
        return self.num_iters
