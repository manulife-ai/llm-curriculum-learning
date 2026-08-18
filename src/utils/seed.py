import random

import numpy as np
import torch

try:
    from transformers import set_seed as hf_set_seed
except ImportError:  # transformers is a runtime dep; guard so seed util stays importable in bare envs.
    hf_set_seed = None


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hf_set_seed is not None:
        hf_set_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
