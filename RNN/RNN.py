import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas
import spacy
from spacy.lang.en import English
from spacy.lang.de import German
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm.auto import tqdm
import random
from collections import Counter
import time
from layers import *
from config import *
import sys
from torch.optim.lr_scheduler import StepLR

def train_step(cfg,model, data, loss_function, optimizer):
    # en_sample = torch.randint(1,8,size = (batch_size,50), device = device)
    # de_sample = torch.randint(1,8,size = (batch_size,51), device = device)
    # model = RNN(cfg).to(device)
    model.train()
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

    loss = loss_function(forwarded.view(-1,50001), de_out_sample.view(-1))
    # print(f'loss = {loss}')
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2.0)
    optimizer.step()
    # print(f'loss.item() = {loss.item()}')
    # print(model.encoder.embedding.weight.grad.size())
    return loss
    

def train(cfg,train_data):
  model = RNN(cfg).to(cfg.device)
  # checkpoint = torch.load(cfg.model_save_path + '/dot_epoch10', map_location = device)
  # model = RNN(cfg)
  # model.load_state_dict(checkpoint['model_state_dict'])
  # del checkpoint
  model.to(device)

  loss_function = nn.CrossEntropyLoss(ignore_index = 0)
  optimizer = optim.SGD(model.parameters(), lr=0.1)
  scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
  batch_size = cfg.batch_size
  # for epoch in range(cfg.num_epochs):
  for epoch in range(cfg.num_epochs):

    '''shuffling data'''
    zipped = list(zip(train_data[0],train_data[1],train_data[2]))
    random.shuffle(zipped)
    total = len(zipped)
    batch_iter = int(total / batch_size)

    for i in tqdm(range(batch_iter), desc = f'epoch = {epoch + 1} // attention = {cfg.attention}'):
      loss = train_step(cfg,model, zipped[i * batch_size: min((i + 1) * cfg.batch_size, total)], loss_function, optimizer)
      if (i % 100) == 0:
        print(f'epoch = {epoch + 1}      batch = {i} / {batch_iter}     loss = {loss}')
        if (i % 1000) == 0:
          torch.save({'epoch' : epoch + 1, 'model_state_dict' : model.state_dict(), 'optimizer_state_dict' : optimizer.state_dict}, f'./results/{cfg.attention}_currentmodel')
    scheduler.step()

    torch.save({'epoch' : epoch + 1, 'model_state_dict' : model.state_dict(), 'optimizer_state_dict' : optimizer.state_dict}, f'./results/{cfg.attention}_epoch{epoch + 1}')
    
      
      

  

  
  
  # scheduler = LambdaLR(optimizer, func)
  

  

  
  




if __name__ == '__main__':
  cfg = Config()
  if torch.cuda.is_available:
  # if 0:
    cfg.device = torch.device("cuda:2")
  else:
    cfg.device = torch.device("cpu")

  print(f'device = {cfg.device}')
  device = cfg.device

  train_data = pickle.load(open('./data/train_indices_tuple', 'rb'))
  print('data loaded')
  train(cfg,train_data)

#   with open(f'./data/en_train', 'r') as f:
#       en = f.readlines()

#   with open(f'./data/de_train', 'r') as f:
#       de = f.readlines()

#   # Setting the number of training sentences we'll use
#   training_examples = 100
#   # We'll be using the spaCy's English and German tokenizers
#   spacy_en = English()
#   spacy_de = German()

#   en_words = Counter()
#   de_words = Counter()
#   en_inputs = []
#   de_inputs = []

#   # Tokenizing the English and German sentences and creating our word banks for both languages
#   for i in tqdm(range(training_examples)):
#       en_tokens = spacy_en(en[i])
#       de_tokens = spacy_de(de[i])
#       if len(en_tokens)==0 or len(de_tokens)==0:
#           continue
#       for token in en_tokens:
#           en_words.update([token.text.lower()])
#       en_inputs.append([token.text.lower() for token in en_tokens] + ['_EOS'])
#       for token in de_tokens:
#           de_words.update([token.text.lower()])
#       de_inputs.append([token.text.lower() for token in de_tokens] + ['_EOS'])

#   # Assigning an index to each word token, including the Start Of String(SOS), End Of String(EOS) and Unknown(UNK) tokens
#   en_words = ['_SOS','_EOS','_UNK'] + sorted(en_words,key=en_words.get,reverse=True)
#   en_w2i = {o:i for i,o in enumerate(en_words)}
#   en_i2w = {i:o for i,o in enumerate(en_words)}
#   de_words = ['_SOS','_EOS','_UNK'] + sorted(de_words,key=de_words.get,reverse=True)
#   de_w2i = {o:i for i,o in enumerate(de_words)}
#   de_i2w = {i:o for i,o in enumerate(de_words)}

#   # Converting our English and German sentences to their token indexes
#   for i in range(len(en_inputs)):
#       en_sentence = en_inputs[i]
#       de_sentence = de_inputs[i]
#       en_inputs[i] = [en_w2i[word] for word in en_sentence]
#       de_inputs[i] = [de_w2i[word] for word in de_sentence]
  

#   hidden_size = 256
#   encoder = EncoderLSTM(len(en_words), hidden_size).to(device)
#   attn = Attention(hidden_size,"concat")
#   decoder = LuongDecoder(hidden_size,len(de_words),attn).to(device)

#   lr = 0.001
#   encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
#   decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)


#   EPOCHS = 10
#   teacher_forcing_prob = 0.5
#   encoder.train()
#   decoder.train()
#   tk0 = tqdm(range(1,EPOCHS+1),total=EPOCHS)
#   for epoch in tk0:
#       avg_loss = 0.
#       tk1 = tqdm(enumerate(en_inputs),total=len(en_inputs),leave=False)
#       for i, sentence in tk1:
#           loss = 0.
#           h = encoder.init_hidden()
#           encoder_optimizer.zero_grad()
#           decoder_optimizer.zero_grad()
#           inp = torch.tensor(sentence).unsqueeze(0).to(device)
#           encoder_outputs, h = encoder(inp,h)

#           #First decoder input will be the SOS token
#           decoder_input = torch.tensor([en_w2i['_SOS']],device=device)
#           #First decoder hidden state will be last encoder hidden state
#           decoder_hidden = h
#           output = []
#           teacher_forcing = True if random.random() < teacher_forcing_prob else False

#           for ii in range(len(de_inputs[i])):
#             decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
#             # Get the index value of the word with the highest score from the decoder output
#             top_value, top_index = decoder_output.topk(1)
#             if teacher_forcing:
#               decoder_input = torch.tensor([de_inputs[i][ii]],device=device)
#             else:
#               decoder_input = torch.tensor([top_index.item()],device=device)
#             output.append(top_index.item())
#             # Calculate the loss of the prediction against the actual word
#             loss += F.nll_loss(decoder_output.view(1,-1), torch.tensor([de_inputs[i][ii]],device=device))
#           loss.backward()
#           encoder_optimizer.step()
#           decoder_optimizer.step()
#           avg_loss += loss.item()/len(en_inputs)
#       tk0.set_postfix(loss=avg_loss)
#     # Save model after every epoch (Optional)
#   torch.save({"encoder":encoder.state_dict(),"decoder":decoder.state_dict(),"e_optimizer":encoder_optimizer.state_dict(),"d_optimizer":decoder_optimizer},"./model.pt")

# encoder.eval()
# decoder.eval()
# # Choose a random sentences
# i = random.randint(0,len(en_inputs)-1)
# h = encoder.init_hidden()
# inp = torch.tensor(en_inputs[i]).unsqueeze(0).to(device)
# encoder_outputs, h = encoder(inp,h)

# decoder_input = torch.tensor([en_w2i['_SOS']],device=device)
# decoder_hidden = h
# output = []
# attentions = []
# while True:
#   decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
#   _, top_index = decoder_output.topk(1)
#   decoder_input = torch.tensor([top_index.item()],device=device)
#   # If the decoder output is the End Of Sentence token, stop decoding process
#   if top_index.item() == de_w2i["_EOS"]:
#     break
#   output.append(top_index.item())
#   attentions.append(attn_weights.squeeze().cpu().detach().numpy())
# print("English: "+ " ".join([en_i2w[x] for x in en_inputs[i]]))
# print("Predicted: " + " ".join([de_i2w[x] for x in output]))
# print("Actual: " + " ".join([de_i2w[x] for x in de_inputs[i]]))

# # Plotting the heatmap for the Attention weights
# fig = plt.figure(figsize=(12,9))
# ax = fig.add_subplot(111)
# cax = ax.matshow(np.array(attentions))
# fig.colorbar(cax)
# ax.set_xticklabels(['']+[en_i2w[x] for x in en_inputs[i]])
# ax.set_yticklabels(['']+[de_i2w[x] for x in output])
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# plt.show()