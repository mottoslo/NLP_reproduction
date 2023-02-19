import torch.nn as nn
import torch
from tqdm.auto import tqdm
from config import *
from preprocess import *
from layers import *
import torch.optim as optim
from util import *
import random
import pickle

def train_step(cfg,model, data, loss_function, optimizer, scheduler, device):
    # en_sample = torch.randint(1,8,size = (batch_size,50), device = device)
    # de_sample = torch.randint(1,8,size = (batch_size,51), device = device)
    # model = RNN(cfg).to(device)
    optimizer.zero_grad()
    en_batch, de_in_batch, de_out_batch = zip(*data)

    en_lst = [torch.tensor(en_line) for en_line in en_batch]
    en_sample = nn.utils.rnn.pad_sequence(en_lst,batch_first = True).to(device)

    de_in_lst = [torch.tensor(de_in_line) for de_in_line in de_in_batch]
    de_in_sample = nn.utils.rnn.pad_sequence(de_in_lst,batch_first = True).to(device)
    
    de_out_lst = [torch.tensor(de_out_line) for de_out_line in de_out_batch]
    de_out_sample = nn.utils.rnn.pad_sequence(de_out_lst,batch_first = True).to(device)

    forwarded = model(en_sample, de_in_sample)
    # print(f'en_sample shape = {en_sample.size()}')
    # print(f'de_in_sample shape = {de_in_sample.size()}')
    # print(f'de_out_sample shape = {de_out_sample.size()}')
    # print(f'en_sample = {min(en_sample.view(-1))}, {max(en_sample.view(-1))}')
    # print(f'de_in_sample = {min(de_in_sample.view(-1))}, {max(de_in_sample.view(-1))}')
    # print(f'de_out_sample = {min(de_out_sample.view(-1))}, {max(de_out_sample.view(-1))}')
    del en_sample
    del de_in_sample

    torch.cuda.empty_cache()
    loss = loss_function(forwarded.view(-1,50001), de_out_sample.view(-1))
    # print(f'loss = {loss}')
    # print(torch.cuda.memory_summary(device=device, abbreviated=False))
    # quit()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2.0)
    optimizer.step()
    scheduler.step()
    # print(f'loss.item() = {loss.item()}')
    # print(model.encoder.embedding.weight.grad.size())
    return loss.item()
    

def train(cfg, model, train_data, loss_function, optimizer, scheduler, device):
  model.train()
  batch_size = cfg.batch_size
  for epoch in range(cfg.num_epochs):

    '''shuffling data'''
    zipped = list(zip(train_data[0],train_data[1],train_data[2]))
    random.shuffle(zipped)
    total = len(zipped)
    batch_iter = int(total / batch_size)

    for i in tqdm(range(batch_iter), desc = f'epoch = {epoch + 1}'):
      loss = train_step(cfg, model, zipped[i * batch_size: min((i + 1) * cfg.batch_size, total)], loss_function, optimizer, scheduler,device)
      if (i % 1000) == 0:
        print(f'epoch = {epoch + 1}      batch = {i} / {batch_iter}     loss = {loss}')
        if (i % 3000) == 0:
          torch.save({'epoch' : epoch + 1, 'model_state_dict' : model.state_dict(), 'optimizer_state_dict' : optimizer.state_dict}, f'./results/trasformer_currentmodel')
        

    torch.save({'epoch' : epoch + 1, 'model_state_dict' : model.state_dict(), 'optimizer_state_dict' : optimizer.state_dict}, f'./results/transformer_epoch{epoch + 1}')




if __name__ == '__main__':
  cfg = Config()
  if torch.cuda.is_available:
  # if 0:
    cfg.device = torch.device("cuda:1")
    device = torch.device("cuda:1")
  else:
    cfg.device = torch.device("cpu")
  print(f'device = {device}')

  model = Transformer(cfg.src_pad_idx, cfg.trg_pad_idx, cfg.trg_sos_idx, cfg.enc_voc_size, cfg.dec_voc_size,
                      cfg.d_model, cfg.n_head, cfg.max_len, cfg.ffn_hidden, cfg.n_layers, cfg.drop_prob, device).to(device)
  train_data = pickle.load(open(cfg.data_folder_path + 'train_indices_tuple', 'rb'))
  loss_function = nn.CrossEntropyLoss(ignore_index = 0)
  optimizer = optim.Adam(model.parameters(), betas = cfg.adam_betas, eps = 1e-09)
  scheduler = TransformerScheduler(optimizer = optimizer, dim_embed = cfg.d_model, warmup_steps = cfg.warmup_steps)
  print('data loaded')
  print(f'transformer model train() starting............. ')
  # print(torch.cuda.memory_summary(device=device, abbreviated=False))
  train(cfg, model, train_data, loss_function, optimizer, scheduler, device)
