from tkinter import Scale
import torch.nn as nn
import torch
import numpy as np
from tqdm.auto import tqdm
from config import *
from preprocess import *
import math
from layers import *
import torch.optim as optim
from util import *


if torch.cuda.is_available:
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

temp_batch_size = 128
temp_vocab_size = 1000
temp_d_model = 512
temp_max_len = 50
temp_drop_prob = 0.2

en_sample = torch.randint(0,8,size = (temp_batch_size,temp_max_len)).to(device)
de_sample = torch.randint(0,8,size = (temp_batch_size,temp_max_len)).to(device)

model = Transformer(0, 0, 3, 50001, 50001, 512, 8, 50, 2048, 4, 0.2, device).to(device)
loss_function = nn.CrossEntropyLoss(ignore_index = 0)
optimizer = optim.Adam(model.parameters(), betas = [0.9,0.98], eps = 1e-09)
scheduler = TransformerScheduler(optimizer = optimizer, dim_embed = temp_d_model, warmup_steps = 4000)
forwarded = model(en_sample, de_sample)

print(f'forwarded_transformer_model = {forwarded.size()}')
print(f'forwarded.view(-1,50001) = {forwarded.view(-1,50001).size()}')
print(f'de_sample.view(-1) = {de_sample.view(-1).size()}')
loss = loss_function(forwarded.view(-1,50001), de_sample.view(-1))
print(f'loss = {loss}')
loss.backward()
optimizer.step()
scheduler.step()





