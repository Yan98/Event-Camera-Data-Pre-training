#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int) -> int:
    return num_seen_examples % buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, mode='ring'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            self.buffer_portion_size = buffer_size 
            
    @torch.no_grad() 
    def init_tensors(self, examples: torch.Tensor) -> None:
        self.examples = torch.zeros((self.buffer_size, *examples.shape[1:]), dtype=examples.dtype, device=examples.device)
    @torch.no_grad()      
    def add_data(self, examples):
        
        if not hasattr(self, 'examples'):
            self.init_tensors(examples)

        for i in range(examples.shape[0]):
            index = ring(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(examples.device)
    @torch.no_grad()              
    def get_data(self, size: int):
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        
        if self.is_empty():
            return None
        
        
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        return self.examples[choice]
 
    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False


    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        delattr(self, "examples")
        self.num_seen_examples = 0
        
m = Buffer(5)
for i in range(10):
    x = torch.ones((1,1)) * i
    m.add_data(x)