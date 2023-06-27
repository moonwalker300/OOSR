import torch
from torch import nn
import numpy as np
from model import weights_init
from torch.nn import functional as F

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.apply(weights_init)
    def forward(self, x, t):
        h = torch.cat((x, t), dim = 1)
        for c in range(self.n_layers):
            if (c == self.n_layers - 1):
                h = self.fc[c](h)
            else:
                h = F.elu(self.fc[c](h))
        h = torch.sigmoid(h)
        return h

def quantile_loss(preds, labels, tau):
    error = labels - preds
    loss_ = torch.max((tau - 1) * error, tau * error)
    return loss_.mean()

def decor_weight(x, t, rs):
    hidden_dim = 10
    n_layers = 3
    cl = Classifier(x.shape[1] + 1, 1, hidden_dim, n_layers)
    epoch = 500
    half_batch_size = 256
    n = x.shape[0]
    bceloss = nn.BCELoss(reduction='mean')
    opt = torch.optim.Adam(cl.parameters(), lr = 0.001, weight_decay = 0)
    for ep in range(epoch):
        idx = np.random.permutation(n)
        tot_loss = 0.0
        for i in range(0, n, half_batch_size):
            op, ed = i, min(i + half_batch_size, n)
            x_batch = torch.FloatTensor(x[idx[op:ed]])
            t_batch = torch.FloatTensor(t[idx[op:ed]])
            t_rand = torch.rand(size = t_batch.size()) * rs
            xx = torch.cat((x_batch, x_batch), dim = 0)
            tt = torch.cat((t_batch, t_rand), dim = 0)
            y = torch.cat((torch.zeros((ed - op, 1)), torch.ones((ed - op, 1))), dim = 0)
            pre = cl(xx, tt)
            loss = bceloss(pre, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += (loss.item() * (ed - op))
        if ((ep + 1) % 100 == 0):
            print('Epoch %d' % (ep))
            print('Loss %f' % (tot_loss / n))
    x_tensor = torch.FloatTensor(x)
    t_tensor = torch.FloatTensor(t)
    pb = cl(x_tensor, t_tensor)
    pb = pb.detach().numpy()
    return pb / (1 - pb)
