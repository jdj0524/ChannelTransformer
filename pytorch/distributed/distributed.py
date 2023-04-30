import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import os

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    