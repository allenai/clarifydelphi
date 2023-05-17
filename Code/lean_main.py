import os
import sys

import torch
import json
import time
import logging
import random
import argparse
import numpy as np
import itertools
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from Code.arguments import get_args
from Code.policy import Policy
from Code.value import Value
from Code.utils.utils import ensure_dir, ceil_div, exact_div, whiten, reduce_mean, reduce_sum, reduce_std, clamp, flatten_dict

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

class PromptDataset(Dataset):
    def __init__(self, path):
        self.prompts = [json.loads(s.strip())["prompt"]["text"].strip() for s in open(path, 'r').readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx]}

class PromptDatasetForDebug(Dataset):
    def __init__(self, situation):
        self.prompts = [situation]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx]}


class PromptCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        prompts = [sequence['prompt'] for sequence in sequences]

        encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return input_ids, attention_mask


class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class AdaptiveKLController:
    def __init__(self, init_kl_coef, params):
        self.value = init_kl_coef
        self.params = params

    def update(self, current, n_steps):
        target = self.params.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.params.horizon
        self.value *= mult
