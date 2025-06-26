# vertical_fl/utils.py
import os
import random
import numpy as np
import torch

GLOBAL_SEED = 42

def set_seed(seed: int):
    # Python built-ins
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (both CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


    # Force single-threaded behavior in PyTorch and BLAS
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
