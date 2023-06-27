import numpy as np
from dataset import Exp_Outcome, Dataset, Parameters, Linear_Outcome
from model import Bootstrap, PolicyNet, PolicyEvaluation, VanillaModel, MCDropoutRegressor, VanillaModel 
import torch
from util import decor_weight
import random
from scipy import stats
import argparse

def RMSE(pre, target):
    mse = np.mean(np.square(pre - target))
    return np.sqrt(mse)

def manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
class Log:
    def __init__(self, filename):
        self.filename = filename
    def log(self, content):
        with open(self.filename, "a") as f:
            f.write(content + '\n')
            f.close()
parser = argparse.ArgumentParser()
parser.add_argument('--lam', type=float, default=10.0, help='Lambda')
parser.add_argument('--tao', type=float, default=0.2, help='Tao')
parser.add_argument('--ifweight', type=int, default=1, help='If Reweight')
parser.add_argument('--samplesize', type=int, default=4000, help='If Reweight')
parser.add_argument('--reexp', type=int, default=10, help='If Reweight')
parser.add_argument('--alpha', type=float, default=4.0, help='If Reweight')
args = parser.parse_args()
lam = args.lam
tao = args.tao
iw = args.ifweight
repeat_exp = args.reexp
alpha = args.alpha
n = args.samplesize
p = 5
rs = 3.0#for exp (3.0 for Exp)
ifnew_param = False
name_param = 'DD'
params = Parameters(p, ifnew_param, name_param)
ifnew_data = True
name_data = 'DD'
data = Dataset(n, p, params, ifnew_data, name_data, rs, alpha)
x_val, t_val, y_val = data.GetValData()

print('---------------Learning Impute Model------------------------')
manual_seed(0)
impute_model = VanillaModel(p, rs)
w_val = decor_weight(x_val, t_val, rs)
impute_model.train(x_val, t_val, y_val, w_val)
y_val = impute_model.GetOutcome(x_val, t_val)
print('---------------Impute Ok-----------------------------')

optim_t = impute_model.search_optimal_t(x_val)
optim_y = impute_model.GetOutcome(x_val, optim_t)
print('Optimal Policy Value:', optim_y.mean())
'''
impute_model = Exp_Outcome(params, rs)
optim_t, optim_y = impute_model.BestTreatmentOutcome(x_val)
print('Optimal Policy Value:', optim_y.mean())
'''
param_list = []
for tao in [0.2, 0.5]:
    for lam in [2.0, 5.0, 10.0, 20.0]:
        param_list.append([lam, tao])
param_list.append([0.0, 1.0])
param_list = np.array(param_list)

best_value = 0.0
best_param = None
fl = Log('Sample_%d_alpha_%f_Exp.txt' % (n, alpha))
for lam, tao in param_list:
    res_tr_list = []
    res_te_list = []
    value_list = []
    print('------------Parameter Select--------------')
    for i in range(0, repeat_exp):
        manual_seed(i)
        if (iw > 0):
            w = decor_weight(x_val, t_val, rs)
            w /= w.mean()
        else:
            w = np.ones([x_val.shape[0], 1])

        reg = MCDropoutRegressor(p, rs, impute_model, lam, tao)
        reg.train_adaptively(x_val, t_val, y_val, impute_model, w)

        y_pre, _ = reg.predict(x_val, t_val)
        print(RMSE(y_pre, y_val))
        print(RMSE(y_pre, impute_model.GetOutcome(x_val, t_val)))
        res_tr_list.append(RMSE(y_pre, impute_model.GetOutcome(x_val, t_val)))
        
        x_test = x_val.copy()
        t_test = np.random.rand(n, 1) * rs
        y_test = impute_model.GetOutcome(x_test, t_test)
        y_pre, _ = reg.predict(x_test, t_test)
        print(RMSE(y_pre, y_test))
        res_te_list.append(RMSE(y_pre, y_test))
        
        dm_t2 = np.zeros([n, 1])
        batch_size = 512
        for i in range(0, n, batch_size):
            op, ed = i, min(n, i + batch_size)
            dm_t2[op:ed] = reg.search_optimal_t(x_val[op:ed])
        dm_y2 = impute_model.GetOutcome(x_val, dm_t2)
        print('DM Search Optimized Policy Value:', dm_y2.mean())
        value_list.append(dm_y2.mean())

    print('Train:', sum(res_tr_list) / len(res_tr_list))
    print('Test:', sum(res_te_list) / len(res_te_list))
    print(value_list)
    mean_value = sum(value_list) / len(value_list)
    print('Param: lam %f tao %f' % (lam, tao))
    fl.log('Param: lam %f tao %f' % (lam, tao))
    print('Value: %f' % mean_value)
    fl.log('Value: %f' % mean_value)
    if (mean_value > best_value):
        best_value = mean_value
        best_param = {'Lam':lam, 'Tao':tao}
print('Best Value:', best_value)
print('Best Param:', best_param)
fl.log('Best Value %f' % best_value)
fl.log('Best Param')
fl.log((str)(best_param))

