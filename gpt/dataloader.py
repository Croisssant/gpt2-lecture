import torch
import os

import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# Load and get batch data
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = master_process

        assert split in {"train", "val"}

        # Get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No Shards found for split {split}"
        if self.master_process:
            print(f"Found {len(shards)} shards for split {split}")
        
        self.reset()
    
    def reset(self):
        # State, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def load_state(self, current_shard, current_position):
        # Restore the position from a checkpoint
        # current_position is rank 0's position (only the master saves it), so offset it for this rank
        self.current_shard = current_shard
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = current_position + self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) 
        y = buf[1:].view(B, T)

        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes

        # If loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y