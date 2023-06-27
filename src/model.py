import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from torch.autograd import Variable

def weights_init(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

class KernelEncoder(nn.Module):
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
    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if (c == self.n_layers - 1):
                h = self.fc[c](h)
            else:
                h = F.elu(self.fc[c](h))
        h = torch.cat((h, torch.ones([h.size()[0], 1])), dim = 1)
        return h

class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, 1)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], 1))
        self.fc = nn.ModuleList(_fc_list)
        self.apply(weights_init)
    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if (c == self.n_layers - 1):
                h = self.fc[c](h)
            else:
                h = F.elu(self.fc[c](h))
        h = torch.sigmoid(h)
        return h

class PolicyEvaluation:
    def __init__(self, band_width):
        self.band_width = band_width
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def gaussian_kernel(self, t, t_tar):
        return torch.exp(-((t - t_tar) / self.band_width) ** 2 / 2) / (self.band_width * np.sqrt(2 * np.pi))
    def IPSestimator(self, x, t, y, ps, t_tar):
        y_tensor = torch.FloatTensor(y).to(self.device)
        ps_tensor = torch.FloatTensor(ps).to(self.device)
        t_tensor = torch.FloatTensor(t).to(self.device)
        tar_pb = self.gaussian_kernel(t_tensor, t_tar)
        return torch.mean((tar_pb / ps_tensor) * y_tensor)
    def SNIPSestimator(self, x, t, y, ps, t_tar):
        y_tensor = torch.FloatTensor(y).to(self.device)
        ps_tensor = torch.FloatTensor(ps).to(self.device)
        t_tensor = torch.FloatTensor(t).to(self.device)
        tar_pb = self.gaussian_kernel(t_tensor, t_tar)
        return ((tar_pb / ps_tensor) * y_tensor).sum() / (tar_pb / ps_tensor).sum()
    def DRestimator(self, x, t, y, ps, t_tar, model):
        x_tensor = torch.FloatTensor(x).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        ps_tensor = torch.FloatTensor(ps).to(self.device)
        t_tensor = torch.FloatTensor(t).to(self.device)
        tar_pb = self.gaussian_kernel(t_tensor, t_tar)
        y_pre = model.predict_gradient(x_tensor, t_tar)
        return torch.mean((tar_pb / ps_tensor) * (y_tensor - y_pre)) + torch.mean(y_pre)
    def DMestimator(self, x, t_tar, model):
        x_tensor = torch.FloatTensor(x).to(self.device)
        return torch.mean(model.predict_gradient(x_tensor, t_tar))

class RegressModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        if (self.n_layers > 1):
            _fc_list = [nn.Linear(self.input_dim + 1, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            self.output_layer = nn.Linear(self.hidden_dim[self.n_layers - 2], 1)
        else:
            _fc_list = []
            self.output_layer = nn.Linear(self.input_dim + 1, 1)
        self.fc = nn.ModuleList(_fc_list)
        self.apply(weights_init)
    def forward(self, x, t):
        h = torch.cat((x, t), dim = 1)
        for c in range(self.n_layers - 1):
            h = F.elu(self.fc[c](h))
        out = self.output_layer(h)
        return out

class VanillaModel:
    def __init__(self, context_dim, rs):
        self.model = RegressModel(context_dim, 100, 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.rs = rs
        self.beta = np.expand_dims(np.linspace(0, 0.5, context_dim), axis = 1)
    def train(self, x, t, y, w):
        n = x.shape[0]
        batch_size = min(n, 512)
        epochs = 100000
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        mse = nn.MSELoss(reduction='mean')
        weight = (w.copy()).squeeze()
        weight /= weight.sum()
        last_loss = 1000000
        current_loss = 0
        last_ep = -1
        intv = (int)(n / batch_size) + 1
        for ep in range(epochs):
            idx = np.random.choice(n, batch_size, replace = True, p = weight)
            x_batch = torch.FloatTensor(x[idx]).to(self.device)
            t_batch = torch.FloatTensor(t[idx]).to(self.device)
            y_batch = torch.FloatTensor(y[idx]).to(self.device)
            pre = self.model(x_batch, t_batch)
            loss = mse(pre, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ((ep + 1) % 1000 == 0):
                print('Epoch %d, Loss %f' % (ep + 1, current_loss))
                current_loss = 0
            current_loss += loss.detach().cpu().item()

    def GetOutcome(self, x, t):
        batch_size = 512
        n = x.shape[0]
        y = np.zeros([n, 1])
        for i in range(0, n, batch_size):
            op, ed = i, min(i + batch_size, n)
            x_batch = torch.FloatTensor(x[op:ed]).to(self.device)
            t_batch = torch.FloatTensor(t[op:ed]).to(self.device)
            y[op:ed] = self.model(x_batch, t_batch).detach().cpu().numpy()
        return y

    def search_optimal_t(self, x):
        n = x.shape[0]
        resolution = 1000
        t = np.zeros([n, 1])
        y = np.zeros([n, 1]) - 10000.0
        for j in range(resolution + 1):
            t_t = np.ones([n, 1]) * j * 1.0 * self.rs / resolution
            pre = self.GetOutcome(x, t_t)
            idx = (pre > y)
            t[idx] = j * 1.0 * self.rs / resolution
            y[idx] = pre[idx]
        return t

class MCDropoutModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, keep_prob = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        if (self.n_layers > 1):
            _fc_list = [nn.Linear(self.input_dim + 1, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            self.output_layer = nn.Linear(self.hidden_dim[self.n_layers - 2], 1)
        else:
            _fc_list = []
            self.output_layer = nn.Linear(self.input_dim + 1, 1)
        self.fc = nn.ModuleList(_fc_list)
        #self.dropout = nn.Dropout(p = 1 - keep_prob)
    def weight_norm(self):
        w = 0
        for name, pa in self.named_parameters():
            if ('weight' in name):
                w += (pa.norm(2) ** 2)
        return w
    def bias_norm(self):
        b = 0
        for name, pa in self.named_parameters():
            if ('bias' in name):
                b += (pa.norm(2) ** 2)
        return b
    def forward(self, x, t):
        h = torch.cat((x, t), dim = 1)
        for c in range(self.n_layers - 1):
            h = F.elu(self.fc[c](h))
            #h = self.dropout(h)
        out = self.output_layer(h)
        return out


class MCDropoutRegressor:
    def __init__(self, context_dim, rs, outcome_model, lam, tao):
        self.kp = 1.0
        self.lamb = 0.0000
        self.rs = rs * 1.0
        self.model = MCDropoutModel(context_dim, 20, 3, self.kp)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.om = outcome_model

        self.lam = lam
        self.tao = tao
    def gau_ker(self, t):
        return np.exp(-t * t / 2) / np.sqrt(2 * np.pi)

    def eval(self, x_val, t_val, y_val, w_val):
        y_pre, _ = self.predict(x_val, t_val)
        return np.mean(np.square(y_pre - y_val) * w_val)

    def train(self, x, t, y, w):
        x_test = x.copy()
        t_test = np.random.rand(x.shape[0], 1) * self.rs
        self.model.apply(weights_init)
        n = x.shape[0]
        batch_size = min(n, 128)
        epochs = 8000000 // n#4000 for TCGAcube#for TCGA1 2000 #8000000 // n#2000 for 4000sample, 4000 for 2000sample
        optimizer = optim.SGD(self.model.parameters(), lr = 0.01, weight_decay = 1e-5)
        mse = nn.MSELoss(reduction='none')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=1e-2)
        weight = (w.copy())
        weight /= weight.mean()

        weight = np.clip(weight, 0.01, 100)
        weight /= weight.mean()

        intv = epochs // 20 #100 for 4000 sample, 200 for 2000 sample
        st_ud = epochs * 2 // 5 #800 for 4000 sample, 1500 for 2000 sample
        tao = self.tao
        lam = self.lam

        last_loss = 1000000.0
        for ep in range(epochs):
            current_loss = 0
            idx = np.random.permutation(n)
            for i in range(0, n, batch_size):
                op, ed = i, min(i + batch_size, n)
                x_batch = torch.FloatTensor(x[idx[op:ed]]).to(self.device)
                t_batch = torch.FloatTensor(t[idx[op:ed]]).to(self.device)
                y_batch = torch.FloatTensor(y[idx[op:ed]]).to(self.device)
                w_batch = torch.FloatTensor(weight[idx[op:ed]]).to(self.device)
                pre = self.model(x_batch, t_batch)
                loss = (mse(pre, y_batch) * w_batch).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current_loss += loss.detach().cpu().item() * (ed - op)
            if ((ep + 1) % 100 == 0):
                print('Epoch %d, Loss %f' % (ep + 1, current_loss / n))

            if (((ep + 1) % intv == 0) and (ep > st_ud)):                
                t_star = self.search_optimal_t(x)
                fz = self.gau_ker((t_star - t) / tao)
                fm = self.norm_constant(t_star, tao)

                '''
                fz = np.exp(self.predict_eval(x, t) / tao)
                fm = self.f_constant(x, tao)
                '''
                weight = (lam * fz / fm + 1) * w
                weight /= weight.mean()
                weight = np.clip(weight, 0.01, 100)
                weight /= weight.mean()
                print((fz/fm).mean(), (fz/fm).std())

                print('ESZ', (np.sum(weight)) ** 2 / np.sum(weight ** 2))
                if (epochs - ep > intv // 2):
                    y_pre, _ = self.predict(x, t)
                    print(np.sqrt(np.mean(np.square(y_pre - self.om.GetOutcome(x, t)))))
                    y_pre, _ = self.predict(x_test, t_test)
                    print(np.sqrt(np.mean(np.square(y_pre - self.om.GetOutcome(x_test, t_test)))))

                    dm_t2 = np.zeros([n, 1])
                    bs = 512
                    for i in range(0, n, bs):
                        op, ed = i, min(n, i + bs)
                        dm_t2[op:ed] = self.search_optimal_t(x[op:ed])
                    dm_y2 = self.om.GetOutcome(x, dm_t2)
                    print('DM Search Optimized Policy Value:', dm_y2.mean())  
                    #self.model.apply(weights_init)
            #scheduler.step()

    def f_constant(self, x, tao, resolution = 1000):
        ret = None
        for i in range(resolution + 1):
            t = np.ones([x.shape[0], 1]) * self.rs * i / resolution
            y_pre = self.predict_eval(x, t)
            if (i == 0):
                ret = np.exp(y_pre / tao)
            else:
                ret += np.exp(y_pre / tao)
        ret /= (resolution + 1)
        return ret        

    def predict_one(self, x, t):
        res = 0
        res2 = 0
        num = 100
        for i in range(num):
            pre = self.model(x, t)
            res += pre.detach().cpu().item()
            res2 += (pre.detach().cpu().item()) ** 2
        return res / num, res2 / num - (res / num) ** 2
    def predict(self, x, t):
        batch_size = 512
        n = x.shape[0]
        y = np.zeros([n, 1])
        u = np.zeros([n, 1])
        for i in range(0, n, batch_size):
            op, ed = i, min(i + batch_size, n)
            x_batch = torch.FloatTensor(x[op:ed]).to(self.device)
            t_batch = torch.FloatTensor(t[op:ed]).to(self.device)
            pre_list = np.zeros((ed - op, 0))
            for j in range(1):
                pre = self.model(x_batch, t_batch)
                pre_list = np.concatenate([pre_list, pre.detach().cpu().numpy()], axis = 1)
            y[op:ed] = np.mean(pre_list, axis = 1, keepdims = True)
            u[op:ed] = 1.96 * np.std(pre_list, axis = 1, keepdims = True)
        return y, u

    def predict_eval(self, x, t):
        batch_size = 512
        n = x.shape[0]
        y = np.zeros([n, 1])
        for i in range(0, n, batch_size):
            op, ed = i, min(i + batch_size, n)
            x_batch = torch.FloatTensor(x[op:ed]).to(self.device)
            t_batch = torch.FloatTensor(t[op:ed]).to(self.device)
            pre = self.model(x_batch, t_batch)
            y[op:ed] = pre.detach().cpu().numpy()
        return y

    def search_optimal_t(self, x):
        n = x.shape[0]
        resolution = 1000
        t = np.zeros([n, 1])
        x_t = torch.FloatTensor(x).to(self.device)
        y = np.zeros([n, 1]) - 10000.0
        for j in range(resolution + 1):
            t_t = (torch.ones((n, 1)) * j * self.rs / resolution).to(self.device)
            pre = self.model(x_t, t_t).detach().cpu().numpy()
            idx = (pre > y)
            t[idx] = j * self.rs / resolution
            y[idx] = pre[idx]
        return t

    def look_response_curve(self, turns, x, outcome_model):
        pre_list = []
        gd_list = []
        ub_list = []
        lb_list = []
        f = open('plot.txt', 'a')
        f.write('Turn %d\n' % (turns))
        for i in range(200 + 1):
            j = (i / 200.0) * self.rs
            tmp_t = np.array([[j]])
            pre, un = self.predict_one(torch.FloatTensor(x).to(self.device),
                                       torch.FloatTensor(tmp_t).to(self.device))
            f.write('%.4f %.4f %.4f %.4f\n' % (pre,
                (outcome_model.GetOutcome(x, tmp_t))[0, 0],
                pre + un * 1.96, pre - un * 1.96))
        f.close()

    def norm_constant(self, t, tao, resolution = 1000):
        ret = np.zeros(t.shape)
        for i in range(resolution + 1):
            pt = np.ones(t.shape) * self.rs * i / resolution
            ret += self.gau_ker((t - pt) / tao)
        ret /= (resolution + 1)
        return ret

    def train_adaptively(self, x, t, y, outcome_model, ww):
        n = x.shape[0]
        w_dbs = ww.copy()
        print(np.square(np.sum(w_dbs)) / (np.sum(w_dbs * w_dbs)))
        self.train(x, t, y, w_dbs)
        #ll = np.array([1068, 757, 480,3993,2773, 735, 2393, 1838, 3230,412, 1543, 3003,3498, 3821, 0,  422, 3118, 3469, 3725,  623])
        #print(t[ll])        
        #for i in range(len(ll)):
        #    self.look_response_curve(0, x[ll[i]:ll[i] + 1], self.om)

class QuantileRegressor:
    def __init__(self, context_dim):
        self.model_ub = MCDropoutModel(context_dim, 20, 3)
        self.model = MCDropoutModel(context_dim, 20, 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_ub.to(self.device)
        self.model.to(self.device)
    def weighted_quantile_loss(self, pre, y, w, tau):
        loss = torch.max(tau * (y - pre), (tau - 1) * (y - pre))
        return (loss * w).mean()
    def train(self, x, t, y, w, tau):
        n = x.shape[0]
        batch_size = min(n, 128)
        epochs = 2000
        params = list(self.model.parameters()) + list(self.model_ub.parameters())
        mse = nn.MSELoss(reduction = 'none')
        optimizer = optim.SGD(params, lr = 0.1, weight_decay = 1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=1e-3)
        weight = (w.copy())
        weight /= weight.mean()
        last_loss = 1000000
        current_loss = 0
        last_ep = -1
        for ep in range(epochs):
            current_loss = 0
            idx = np.random.permutation(n)
            for i in range(0, n, batch_size):
                op, ed = i, min(i + batch_size, n)
                x_batch = torch.FloatTensor(x[idx[op:ed]]).to(self.device)
                t_batch = torch.FloatTensor(t[idx[op:ed]]).to(self.device)
                y_batch = torch.FloatTensor(y[idx[op:ed]]).to(self.device)
                w_batch = torch.FloatTensor(weight[idx[op:ed]]).to(self.device)
                pre_ub = self.model_ub(x_batch, t_batch)
                pre = self.model(x_batch, t_batch)
                loss = self.weighted_quantile_loss(pre, y_batch, w_batch, 1 - tau) + \
                        self.weighted_quantile_loss(pre_ub, y_batch, w_batch, tau)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current_loss += loss.item() * (ed - op)
            if ((ep + 1) % 200 == 0):
                print('Epoch %d, Loss %f' % (ep + 1, current_loss / n))
            scheduler.step()
            if (last_loss > current_loss):
                last_loss = current_loss
                last_ep = ep
            if (ep - last_ep > 5000):
                break
    def predict_one(self, x, t):
        pre = self.model(x, t)
        pre_ub = self.model_ub(x, t)
        return (pre.item() + pre_ub.item()) / 2, (pre_ub.item() - pre.item()) / 2

    def predict(self, x, t):
        batch_size = 512
        n = x.shape[0]
        y = np.zeros([n, 1])
        u = np.zeros([n, 1])
        for i in range(0, n, batch_size):
            op, ed = i, min(i + batch_size, n)
            x_batch = torch.FloatTensor(x[op:ed]).to(self.device)
            t_batch = torch.FloatTensor(t[op:ed]).to(self.device)
            pre = self.model(x_batch, t_batch).detach().cpu().numpy()
            pre_ub = self.model_ub(x_batch, t_batch).detach().cpu().numpy()
            y[op:ed] = (pre_ub + pre) / 2
            u[op:ed] = (pre_ub - pre) / 2
        return y, u


    def search_optimal_t(self, x):
        n = x.shape[0]
        resolution = 1000
        t = np.zeros([n, 1])
        x_t = torch.FloatTensor(x).to(self.device)
        y = np.zeros([n, 1]) - 10000.0
        for j in range(resolution + 1):
            t_t = (torch.ones((n, 1)) * j * 1.0 / resolution).to(self.device)
            pre = (self.model(x_t, t_t) + self.model_ub(x_t, t_t)).detach().cpu().numpy()
            idx = (pre > y)
            t[idx] = j * 1.0 / resolution
            y[idx] = pre[idx]
        return t

    def look_response_curve(self, turns, x, outcome_model):
        pre_list = []
        gd_list = []
        ub_list = []
        lb_list = []
        f = open('plot.txt', 'a')
        f.write('Turn %d\n' % (turns))
        for i in range(200 + 1):
            j = i / 200.0
            tmp_t = np.array([[j]])
            pre, un = self.predict_one(torch.FloatTensor(x).to(self.device),
                                       torch.FloatTensor(tmp_t).to(self.device))
            f.write('%.4f %.4f %.4f %.4f\n' % (pre,
                (outcome_model.GetOutcome(x, tmp_t))[0, 0],
                pre + un, pre - un))
        f.close()

    def norm_constant(self, x, tao, resolution = 50):
        ret = None
        for i in range(resolution + 1):
            t = np.ones([x.shape[0], 1]) * 1.0 * i / resolution
            y_pre, u_pre = self.predict(x, t)
            if (i == 0):
                ret = np.exp((y_pre + u_pre) / tao)
            else:
                ret += np.exp((y_pre + u_pre) / tao)
        ret /= (resolution + 1)
        return ret

    def train_adaptively(self, x, t, y, outcome_model, ww):
        iters = 1
        n = x.shape[0]
        tau = 0.1
        tao = 0.6
        intv = (tau - 0.5) / iters
        w_at = np.ones([n, 1])
        for i in range(iters):
            w_now = w_at * ww
            w_now /= w_now.mean()
            print(np.square(w_now.sum()) / (w_now * w_now).sum())
            self.train(x, t, y, w_now, tau)
            y_pre, u_pre = self.predict(x, t)
            w_new = np.exp((y_pre + u_pre) / tao)
            fm = self.norm_constant(x, tao, 50)
            w_new /= fm
            w_new /= w_new.mean()
            w_at = w_new
            tau -= intv
            for j in range(100, 110):
                self.look_response_curve(i, x[j:j + 1], outcome_model)
                
class Bootstrap:
    def __init__(self, context_dim, model_num):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_list = []
        self.idx_list = []
        for i in range(model_num):
            self.model_list.append(RegressModel(context_dim, 20, 3))
            self.model_list[i].to(self.device)
        self.model_num = model_num
    def train(self, x, t, y, w):
        n = x.shape[0]
        batch_size = min(n, 128)
        for i in range(self.model_num):
            print('Model %d' % (i))
            epochs = 2000
            self.model_list[i].apply(weights_init)
            optimizer = optim.Adam(self.model_list[i].parameters(), lr=0.01)
            mse = nn.MSELoss(reduction='none')
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=1e-3)
            if (len(self.idx_list) < self.model_num):
                n_sel = n
                select_idx = np.random.choice(n, n_sel, replace = True)
                self.idx_list.append(select_idx)
            else:
                select_idx = self.idx_list[i]
                n_sel = np.shape(select_idx)[0]

            x_sel = x[select_idx]
            t_sel = t[select_idx]
            y_sel = y[select_idx]
            w_sel = w[select_idx].copy()
            w_sel /= w_sel.mean()
            last_loss = 1000000
            last_ep = -1
            current_loss = 0
            for ep in range(epochs):
                idx = np.random.permutation(n_sel)
                current_loss = 0
                for j in range(0, n_sel, batch_size):
                    op, ed = j, min(j + batch_size, n_sel)
                    x_batch = torch.FloatTensor(x_sel[idx[op:ed]]).to(self.device)
                    t_batch = torch.FloatTensor(t_sel[idx[op:ed]]).to(self.device)
                    y_batch = torch.FloatTensor(y_sel[idx[op:ed]]).to(self.device)
                    w_batch = torch.FloatTensor(w_sel[idx[op:ed]]).to(self.device)
                    pre = self.model_list[i](x_batch, t_batch)
                    loss = (mse(pre, y_batch) * w_batch).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    current_loss += loss.item() * (ed - op)
                if ((ep + 1) % 200 == 0):
                    print('Epoch %d, Loss %f' % (ep + 1, current_loss / n))
                #scheduler.step()
                if (last_loss > current_loss):
                    last_loss = current_loss
                    last_ep = ep
                if (ep - last_ep > 5000):
                    break

    def norm_constant(self, x, mu, tao, resolution = 50):
        ret = None
        for i in range(resolution + 1):
            t = np.ones([x.shape[0], 1]) * 1.0 * i / resolution
            y_pre, u_pre = self.predict(x, t)
            if (i == 0):
                ret = np.exp((y_pre + mu * u_pre) / tao)
            else:
                ret += np.exp((y_pre + mu * u_pre) / tao)
        ret /= (resolution + 1)
        return ret

    def train_adaptively(self, x, t, y, outcome_model, ww):
        iters = 1
        n = x.shape[0]
        w = ww.copy()
        mu = 1.96
        tao = 0.6
        for i in range(iters):
            print(np.square(w.sum()) / (w * w).sum())
            self.train(x, t, y, w)
            y_pre, u_pre = self.predict(x, t)
            w_new = np.exp((y_pre + u_pre * mu) / tao)
            fm = self.norm_constant(x, mu, tao, 50)
            w_new /= fm
            w_new /= w_new.mean()
            w = w_new
            mu *= 0.8
            for j in range(10, 15):
                self.look_response_curve(i, x[j:j + 1], outcome_model)
    def predict_one(self, x, t):
        res = 0
        res2 = 0
        for i in range(self.model_num):
            pre = self.model_list[i](x, t)
            res += pre.detach().cpu().item()
            res2 += (pre.detach().cpu().item()) ** 2
        return res / self.model_num, res2 / self.model_num - (res / self.model_num) ** 2

    def predict_gradient(self, x, t):
        res = None
        for i in range(self.model_num):
            pre = self.model_list[i](x, t)
            res = pre if (i == 0) else (res + pre)
        res /= self.model_num
        return res

    def look_response_curve(self, turns, x, outcome_model):
        pre_list = []
        gd_list = []
        ub_list = []
        lb_list = []
        f = open('plot.txt', 'a')
        f.write('Turn %d\n' % (turns))
        for i in range(200 + 1):
            j = i / 200.0
            tmp_t = np.array([[j]])
            pre, un = self.predict_one(torch.FloatTensor(x).to(self.device),
                                       torch.FloatTensor(tmp_t).to(self.device))
            f.write('%.4f %.4f %.4f %.4f\n' % (pre, 
                (outcome_model.GetOutcome(x, tmp_t))[0, 0], 
                pre + un * 10, pre - un * 10))
        f.close()
    def predict(self, x, t):
        batch_size = 512
        n = x.shape[0]
        y = np.zeros([n, 1])
        u = np.zeros([n, 1])
        for i in range(0, n, batch_size):
            op, ed = i, min(i + batch_size, n)
            x_batch = torch.FloatTensor(x[op:ed]).to(self.device)
            t_batch = torch.FloatTensor(t[op:ed]).to(self.device)
            pre_list = np.zeros((ed - op, 0))

            for j in range(self.model_num):
                pre = self.model_list[j](x_batch, t_batch)
                pre_list = np.concatenate([pre_list, pre.detach().cpu().numpy()], axis = 1)
            y[op:ed] = np.mean(pre_list, axis = 1, keepdims = True)
            u[op:ed] = np.std(pre_list, axis = 1, keepdims = True)
        return y, u

    def search_optimal_t(self, x):
        n = x.shape[0]
        resolution = 1000
        t = np.zeros([n, 1])
        x_t = torch.FloatTensor(x).to(self.device)
        y = np.zeros([n, 1]) - 10000.0
        for j in range(resolution + 1):
            t_t = (torch.ones((n, 1)) * j * 1.0 / resolution).to(self.device)
            pre = self.predict_gradient(x_t, t_t).detach().cpu().numpy()
            idx = (pre > y)
            t[idx] = j * 1.0 / resolution
            y[idx] = pre[idx]
        return t
