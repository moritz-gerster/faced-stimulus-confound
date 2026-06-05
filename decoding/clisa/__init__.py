"""CLISA package (Contrastive Learning for Inter-Subject Alignment)."""
import random
import numpy as np
import torch

from decoding.config import RANDOM_SEED


def _init_torch():
    """Shared seed and CUDA device setup for reproducibility."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)
    device = torch.device('cuda')
    torch.cuda.set_device(0)
    return device
