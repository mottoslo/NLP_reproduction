import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
import random
from collections import Counter
import time
from config import *
import pickle

class Attention(nn.Module):
  def __init__(self, cfg):
    super(Attention, self).__init__()
    self.device = cfg.device
    self.method = cfg.attention
    self.hidden_size = cfg.lstm_hidden_size
    self.batch_size = cfg.batch_size

    # Defining the layers/weights required depending on alignment scoring method
    if self.method == "general":
      self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    elif self.method == "concat":
      self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
      self.weight = nn.Parameter(torch.FloatTensor(cfg.max_seq_len, self.hidden_size)).to(self.device)
      # self.weight = nn.Parameter(torch.FloatTensor(cfg.max_seq_len, self.hidden_size))

  def forward(self, decoder_outputs, encoder_outputs):
    if self.method == "dot":
      # For the dot scoring method, no weights or linear layers are involved
      # print(f'encoder_outputs shape = {encoder_outputs.size()}')
      # print(f'decoder outputs shape = {decoder_outputs.transpose(1,2).size()}')
      # print(f'bmmed shape = {encoder_outputs.bmm(decoder_outputs.transpose(1,2))}')
      return encoder_outputs.bmm(decoder_outputs.transpose(1,2))


    elif self.method == "general":
      # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
      out = self.fc(decoder_outputs)
      # print(f'out shape = {out.size()}')
      return encoder_outputs.bmm(out.transpose(1,2))

    ''''''''''''''''''''''''''''''#concat 구현해야함''''''''''''''''''''''''''''''''''''''''''
    # elif self.method == "concat":
    #   # For concat scoring, decoder hidden state and encoder outputs are concatenated first
    #   out = torch.tanh(self.fc(decoder_outputs + encoder_outputs))
    #   print(f'out shape = {out.size()}')
    #   print(f'weight unsqueeze shape = {self.weight.size()}')
    #   return out.bmm(self.weight)

class EncoderLSTM(nn.Module,):
  def __init__(self, cfg):
    super(EncoderLSTM, self).__init__()
    self.hidden_size = cfg.lstm_hidden_size
    self.n_layers = cfg.lstm_n_layers
    self.vocab_size = cfg.max_vocab_size
    self.drop_prob = cfg.lstm_drop_prob
    self.device = cfg.device

    self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx = 0)
    self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.drop_prob, batch_first=True)

  def forward(self, batch_input, hidden):
    # Embed input words
    embedded = self.embedding(batch_input)
    # Pass the embedded word vectors into LSTM and return all outputs
    output, hidden = self.lstm(embedded, hidden)
    return output, hidden

  def init_hidden(self, batch_size=32):
    return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device))


class LuongDecoder(nn.Module):
  def __init__(self, cfg):
    super(LuongDecoder, self).__init__()
    self.hidden_size = cfg.lstm_hidden_size
    self.output_size = cfg.max_vocab_size
    self.n_layers = cfg.lstm_n_layers
    self.drop_prob = cfg.lstm_drop_prob

    # The Attention Mechanism is defined in a separate class
    self.attention = Attention(cfg)

    self.embedding = nn.Embedding(self.output_size, self.hidden_size, padding_idx = 0)
    self.dropout = nn.Dropout(self.drop_prob)
    self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.drop_prob, batch_first=True)

  def forward(self, batch_input, hidden, encoder_outputs):
    # Embed input words
    embedded = self.embedding(batch_input)
    embedded = self.dropout(embedded)

    # Passing previous output word (embedded) and hidden state into LSTM cell
    lstm_out, hidden = self.lstm(embedded, hidden)        #decoder_hidden = (num_layers, batch, dim)
    # return lstm_out, hidden

    # Calculating Alignment Scores - see Attention class for the forward pass function
    alignment_scores = self.attention(lstm_out, encoder_outputs)     # (num_layers,batch,batch)

    # Softmaxing alignment scores to obtain Attention weights

    attn_weights = F.softmax(alignment_scores, dim=1).transpose(1,2)

    # Multiplying Attention weights with encoder outputs to get context vector
    context_vector = torch.bmm(attn_weights,encoder_outputs)

    # Concatenating output from LSTM with context vector
    output = torch.cat((lstm_out, context_vector),-1)

    return output, hidden, attn_weights


class RNN(nn.Module):
  def __init__(self, cfg):
    super(RNN, self).__init__()
    self.cfg = cfg
    self.encoder = EncoderLSTM(cfg)
    self.decoder = LuongDecoder(cfg)
    self.classifier = nn.Linear(cfg.lstm_hidden_size * 2, cfg.max_vocab_size)

  def forward(self, lang1_batch_input, lang2_batch_input):
    hidden1 = self.encoder.init_hidden(batch_size = self.cfg.batch_size)
    # encoder_output, encoder_hidden = self.encoder(lang1_batch_input, hidden1)
    encoder_output, _ = self.encoder(lang1_batch_input, hidden1)

    # decoder_output, decoder_hidden, attn_weights = decoder(lang2_batch_input, hidden1, encoder_outputs = encoder_output)
    decoder_output, _, _ = self.decoder(lang2_batch_input, hidden1, encoder_outputs = encoder_output)

    # Pass concatenated vector through Linear layer acting as a Classifier
    # output = F.log_softmax(self.classifier(decoder_output), dim=1).argmax(dim = 2)
    # output = F.log_softmax(self.classifier(decoder_output), dim=1)
    output = self.classifier(decoder_output)
    return output

    

    


    



if __name__ == '__main__':
    cfg = Config()
    # if torch.cuda.is_available:
    if torch.cuda.is_available:
        device = torch.device("cuda")
        cfg.device = device
    else:
        device = torch.device("cpu")
    print(f'device = {device}')

    # '''checking Encoder'''    
    # batch_size = 32

    # en_sample = torch.randint(1,8,size = (batch_size,50), device = device)


    # encoder = EncoderLSTM(cfg).to(device)
    # hidden1 = encoder.init_hidden(batch_size = batch_size)
    # encoder_output, encoder_hidden = encoder(en_sample, hidden1)
    
    # print(f'Encoder output = {encoder_output.size()}')
    # print(f'Encoder h_n = {encoder_hidden[0].size()}')
    # print(f'Encoder c_n = {encoder_hidden[1].size()}')

    # '''checking Decoder'''    
    # de_sample = torch.randint(1,8,size = (batch_size,50), device = device)
    # decoder = LuongDecoder(cfg).to(device)
    # decoder_output, decoder_hidden, attn_weights = decoder(de_sample, hidden1, encoder_outputs = encoder_output)

    # print(f'Decoder hidden[0] = {decoder_hidden[0].size()}')
    # print(f'Decoder hidden[1] = {decoder_hidden[1].size()}')
    # print(f'attention weights = {attn_weights.size()}')
    # print(f'Decoder output = {decoder_output.size()}')
    # print(f'Decoder output label = {decoder_output.argmax(dim = 2)}')
    # '''Checking Attention'''

    # attention = Attention(cfg)
    # score = attention(decoder_output, encoder_output)
    # weights = F.softmax(score, dim = 0)
    # print(f'Attention score = {score.size()}')
    # print(f'Attention weights = {weights.size()}')
    # print(f'Attention weights columnwise sum = {weights.sum(axis = 0)}')

    '''Checking RNN'''
    # batch_size = cfg.batch_size
    # en_sample = torch.randint(1,8,size = (batch_size,30), device = device)
    # de_sample = torch.randint(1,8,size = (batch_size,36), device = device)
    train_data = pickle.load(open(f'./data/train_indices_tuple_small', 'rb'))
    en_train = train_data[0]
    de_in_train = train_data[1]
    de_out_train = train_data[2]
    batch_size = cfg.batch_size

    lst = [torch.tensor(en_train[i]) for i in range(batch_size)]
    en_sample = nn.utils.rnn.pad_sequence(lst,batch_first = True).to(device)

    lst = [torch.tensor(de_in_train[i]) for i in range(batch_size)]
    de_sample = nn.utils.rnn.pad_sequence(lst,batch_first = True).to(device)
    
    lst = [torch.tensor(de_out_train[i]) for i in range(batch_size)]
    de_out_sample = nn.utils.rnn.pad_sequence(lst,batch_first = True).to(device)
    
    model = RNN(cfg).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    forwarded = model(en_sample,de_sample)
    print(f'weight before optimizer = {model.encoder.embedding.weight.grad}')
    
    print(f'en_sample = {en_sample.size()}')
    print(f'de_in_sample = {de_sample.size()}')
    print(f'de_out_sample = {de_out_sample.size()}')
    print(f'forwarded = {forwarded.size()}')
    print(f'forwarded_changed = {forwarded.view(-1,50001).size()}')
    print(f'de_sample_changed = {de_out_sample.view(-1).size()}')

    forwarded_classes = forwarded.argmax(dim = 2)
    print(f'forwarded_classes shape = {forwarded_classes.size()}')

    # print(f'encoder emb weight = {model.encoder.embedding.weight}')

    loss = loss_function(forwarded.view(-1,50001), de_out_sample.view(-1))
    loss.backward()
    optimizer.step()
    # print(f'encoder emb weight after = {model.encoder.embedding.weight}')

    print(loss)
    



    