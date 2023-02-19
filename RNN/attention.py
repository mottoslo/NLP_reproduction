import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dotproductattention(h_t, h_s): #(batch * hidden), (batch * hidden * length) 
    b, h = h_t.shape
    h_t = h_t.view(b, 1, h)
    score = torch.bmm(h_t, h_s).squeeze(1) #ToDo (batch * 1 * length) 
    a_v = nn.Softmax(score, dim=1) #(batch * length)
    return a_v

def generalattention(h_t, h_s, W_a): #(batch * hidden), (batch * hidden * length), (hidden * hidden)
    b, h = h_t.shape
    temp = torch.mm(h_t, W_a).view(b, 1, h)
    score = torch.bmm(temp, h_s).squeeze(1)
    a_v = nn.Softmax(score, dim=1)
    return a_v

def concatattention(h_t, h_s, W_a, v_a): #(batch * hidden), (batch * hidden * length), (2*hidden, hidden), (hidden, 1)
    b, h, l= h_s.shape
    h_t = h_t.tile(1, 1, l) #(batch, hidden, length)
    h_s.cat(h_t, dim=1) #(batch, 2*hidden, length)
    h_s = h_s.view(b, l, -1) #batch, length, 2*hidden
    score = torch.mm(h_s, W_a) #batch, length, hidden
    score = torch.mm(score, v_a).squeeze() #Batch, length
    a_v = nn.Softmax(score, dim=1)
    return a_v

def locationattention(h_t, W_a): #(batch, hidden), (hidden, hidden):
    score = torch.mm(h_t, W_a)
    a_v = nn.Softmax(score, dim=1)
    return a_v

class Attention(nn.Module):
    def __init__(self, score, hidden_dim, max_length, att_type='global'):
        super(Attention, self).__init__()
        self.score = score
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.att_type = att_type
        self.window_size = 10

        if score == 'general':
            self.att_score = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif score == 'concat':
            self.att_score = nn.Linear(2*hidden_dim, hidden_dim, bias=False)
            self.v = nn.parameter(torch.FloatTensor(hidden_dim, 1))
        elif score == 'location':
            self.att_score = nn.Linear(hidden_dim, max_length, bias=False)

    def forward(self, h_t, h_s, device): #(batch, length, hidden), (batch, max_length, hidden)
        b, l, h = h_s.shape
        # h_t = h_t.view(b, 1, h)
        if self.score == 'dot':
            score = torch.bmm(h_t, h_s.transpose(2, 1)) #(batch, length, max_length) 
        elif self.score == 'general':
            score = self.att_score(h_s) #(b, max_l ,h)
            score = torch.bmm(h_t, score.transpose(2, 1)) #(b, l, max_l)
        elif self.score == 'concat':
            # h_t = h_t.tile(1, l, 1) #(batch, length, hidden)
            h_s.cat(h_t, dim=2) #(batch, length, 2*hidden)
            score = self.att_score(h_s) #batch, max_length, hidden
            score = torch.tanh(score)
            score = torch.mm(score, self.v).transpose(2, 1)
        elif self.score == 'location':
            score = self.att_score(h_t) #batch, length, max_length
        # score = score.squeeze(1)
        # score = F.softmax(score, dim=1) #b, l

        if self.att_type == 'local_m':
            loc = torch.zeros(l, self.max_length, device=device)
            for i in range(l):
                left = max(0, i-self.window_size)
                right = min(self.max_length, i+self.window_size+1)
                loc[i, left:right] = torch.ones(right-left)
            score *= loc
            score[score==0] -= 1e+06

        score = F.softmax(score, dim=2) #b, length, max_length
        return score