import numpy as np
from dataset import Exp_Outcome, Dataset, Parameters, Linear_Outcome, Logit_Outcome
from model import Bootstrap, PolicyNet, PolicyEvaluation, VanillaModel, MCDropoutRegressor, QuantileRegressor
import torch
from util import decor_weight
import random
from scipy import stats
import argparse
import os
def RMSE(pre, target):
    mse = np.mean(np.square(pre - target))
    return np.sqrt(mse)

class Log:
    def __init__(self, filename):
        self.filename = filename
    def log(self, content):
        with open(self.filename, "a") as f:
            f.write(content + '\n')
            f.close()
        #print(content)
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,5,7'
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
rs = 3.0
ifnew_param = False
name_param = 'DD'
params = Parameters(p, ifnew_param, name_param)
ifnew_data = True
name_data = 'DD'
data = Dataset(n, p, params, ifnew_data, name_data, rs, alpha)
x, t, y, ps = data.GetData()
x_tes = data.GetTestData()
outcome_model = Exp_Outcome(params, rs)

print('--------------------------------------------------------------------------')
print(args)
print('6324 10000 Testing')
print('Model 2 hidden layers 20 units!') #default 2 hidden layers 20 units
print(t.mean(), t.std())
print(y.mean(), y.std())
print('Inverse Propensity Score Weight STD and Mean', np.std(1 / ps), np.mean(1 / ps))
print((1 / ps).max())
def manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

optim_t, optim_y = outcome_model.BestTreatmentOutcome(x)
print('Optimal Policy Value:', optim_y.mean())
optim_t_tes, optim_y_tes = outcome_model.BestTreatmentOutcome(x_tes)
print('Optimal Policy Value for Test', optim_y_tes.mean())

res_tr_list = []
res_te_list = []
value_list = []
value_tes_list = []
rho_star_list = []
rho_f_list = []
for i in range(0, repeat_exp):
    manual_seed(i)
    if (iw > 0):
        w = decor_weight(x, t, rs)
        w /= w.mean()
    else:
        w = np.ones([x.shape[0], 1])

    reg = MCDropoutRegressor(p, rs, outcome_model, lam, tao)
    reg.train_adaptively(x, t, y, outcome_model, w)

    y_pre, _ = reg.predict(x, t)
    print(RMSE(y_pre, y))
    print(RMSE(y_pre, outcome_model.GetOutcome(x, t)))
    res_tr_list.append(RMSE(y_pre, outcome_model.GetOutcome(x, t)))

    x_test = x.copy()
    t_test = np.random.rand(n, 1) * rs
    y_test = outcome_model.GetOutcome(x_test, t_test)
    y_pre, _ = reg.predict(x_test, t_test)
    print(RMSE(y_pre, y_test))
    res_te_list.append(RMSE(y_pre, y_test))
    
    dm_t2 = np.zeros([n, 1])
    batch_size = 512
    for i in range(0, n, batch_size):
        op, ed = i, min(n, i + batch_size)
        dm_t2[op:ed] = reg.search_optimal_t(x[op:ed])
    dm_y2 = outcome_model.GetOutcome(x, dm_t2)
    print('DM Search Optimized Policy Value:', dm_y2.mean())
    gap = (optim_y - dm_y2).squeeze()
    idx = gap.argsort()
    print(idx[-20:])
    #print(np.concatenate([optim_y[idx[-20:]], dm_y2[idx[-20:]]], axis = 1))
    print('Over optimistic', gap[(dm_t2.squeeze() > rs - 0.001) | (dm_t2.squeeze() < 0.001)].sum() / n)
    value_list.append(dm_y2.mean())

    dm_t2_tes = np.zeros([x_tes.shape[0], 1])
    for i in range(0, x_tes.shape[0], batch_size):
        op, ed = i, min(x_tes.shape[0], i + batch_size)
        dm_t2_tes[op:ed] = reg.search_optimal_t(x_tes[op:ed])
    dm_y2_tes = outcome_model.GetOutcome(x_tes, dm_t2_tes)
    print('DM Search Optimized Policy Value for Test:', dm_y2_tes.mean())
    value_tes_list.append(dm_y2_tes.mean())
    pre_rho_f, _ = reg.predict(x_tes, dm_t2_tes)
    pre_rho_star, _ = reg.predict(x_tes, optim_t_tes)
    rho_f_list.append(RMSE(pre_rho_f, dm_y2_tes))
    rho_star_list.append(RMSE(pre_rho_star, optim_y_tes))
    print('Rho f:', RMSE(pre_rho_f, dm_y2_tes))
    print('Rho star:', RMSE(pre_rho_star, optim_y_tes))

print('Train:', sum(res_tr_list) / len(res_tr_list))
print('Test:', sum(res_te_list) / len(res_te_list))
print(value_list)
print(value_tes_list)
value_list = np.array(value_list)
value_tes_list = np.array(value_tes_list)
rho_f_list = np.array(rho_f_list)
rho_star_list = np.array(rho_star_list)
print('Value Mean:%f, Std: %f' % (value_list.mean(), value_list.std()))
print('Value Test Mean:%f, Std: %f' % (value_tes_list.mean(), value_tes_list.std()))
print('Rho f Mean:%f, Std %f' % (rho_f_list.mean(), rho_f_list.std()))
print('Rho Star Mean:%f, Std %f' %(rho_star_list.mean(), rho_star_list.std()))
exit(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

policy_eva = PolicyEvaluation(0.2)
t_tensor = torch.FloatTensor(optim_t).to(device)

l1 = []
l2 = []
l3 = []
l4 = []
for it in range(20):
    print('IPS Policy Value:', policy_eva.IPSestimator(x, t, y, ps, t_tensor))
    print('SNIPS Policy Value:', policy_eva.SNIPSestimator(x, t, y, ps, t_tensor))
    print('DM Policy Value:', policy_eva.DMestimator(x, t_tensor, reg))
    print('DR Policy Value:', policy_eva.DRestimator(x, t, y, ps, t_tensor, reg))
    l1.append(policy_eva.IPSestimator(x, t, y, ps, t_tensor).detach().cpu().item())
    l2.append(policy_eva.SNIPSestimator(x, t, y, ps, t_tensor).detach().cpu().item())
    l3.append(policy_eva.DMestimator(x, t_tensor, reg).detach().cpu().item())
    l4.append(policy_eva.DRestimator(x, t, y, ps, t_tensor, reg).detach().cpu().item())

print(sum(l1) / len(l1))
print(sum(l2) / len(l2))
print(sum(l3) / len(l3))
print(sum(l4) / len(l4))

# print(optim_y)
# print(dm_y2)
# print(optim_t)
# print(dm_t2)
# print((optim_y > dm_y2).sum())
# print((optim_y < dm_y2).sum())
# for pick in range(n):
#     pre_list = []
#     gd_list = []
#     x_list = []
#     #pick = (int)(input('Input:'))
#     tmp_x = x[pick:pick + 1]
#     _, yy = outcome_model.BestTreatmentOutcome(tmp_x)
#     print(outcome_model.BestTreatmentOutcome(tmp_x))
#     tt = reg.search_optimal_t(tmp_x)
#     print(tt, outcome_model.GetOutcome(tmp_x, tt))
#     if (yy > outcome_model.GetOutcome(tmp_x, tt) + 0.5):
#         for i in range(200 + 1):
#             j = i / 200.0
#             tmp_t = np.array([[j]])
#             pre = reg.predict_gradient(torch.FloatTensor(tmp_x), torch.FloatTensor(tmp_t))
#             pre_list.append(pre.item())
#             gd_list.append((outcome_model.GetOutcome(tmp_x, tmp_t))[0, 0])
#             x_list.append(j)
#
#
#         plt.plot(x_list, pre_list, color='red', label = 'Predict')
#         plt.plot(x_list, gd_list, color='blue', label = 'Ground Truth')
#         plt.show()

l1 = []
l2 = []
l3 = []
l4 = []
for it in range(10):
    # policy_ips = PolicyNet(p, 10, 3)
    # optimizer = torch.optim.Adam(policy_ips.parameters(), lr = 0.001)
    # iter_ips = 20000
    # x_tensor = torch.Tensor(x)
    # for it in range(iter_ips):
    #     t_tar = policy_ips(x_tensor)
    #     utility = -policy_eva.IPSestimator(x, t, y, ps, t_tar)
    #     if ((it + 1) % (iter_ips / 50) == 0):
    #         print(utility)
    #     optimizer.zero_grad()
    #     utility.backward()
    #     optimizer.step()
    # t_tar = policy_ips(x_tensor)
    # t_tar = t_tar.detach().cpu().numpy()
    # print('IPS Optimized Policy Value:', np.mean(outcome_model.GetOutcome(x, t_tar)))
    # l1.append(np.mean(outcome_model.GetOutcome(x, t_tar)))

    # policy_snips = PolicyNet(p, 10, 3)
    # optimizer = torch.optim.Adam(policy_snips.parameters(), lr = 0.001)
    # iter_snips = 20000
    # x_tensor = torch.Tensor(x)
    # for it in range(iter_snips):
    #     t_tar = policy_snips(x_tensor)
    #     utility = -policy_eva.SNIPSestimator(x, t, y, ps, t_tar)
    #     if ((it + 1) % (iter_snips / 50) == 0):
    #         print(utility)
    #     optimizer.zero_grad()
    #     utility.backward()
    #     optimizer.step()
    # t_tar = policy_snips(x_tensor)
    # t_tar = t_tar.detach().cpu().numpy()
    # print('SNIPS Optimized Policy Value:', np.mean(outcome_model.GetOutcome(x, t_tar)))
    # l2.append(np.mean(outcome_model.GetOutcome(x, t_tar)))

    policy_dm = PolicyNet(p, 10, 3)
    optimizer = torch.optim.Adam(policy_dm.parameters(), lr = 0.001)
    iter_dm = 8000
    x_tensor = torch.Tensor(x)
    for it in range(iter_dm):
        t_tar = policy_dm(x_tensor)
        utility = -policy_eva.DMestimator(x, t_tar, reg)
        if ((it + 1) % (iter_dm / 50) == 0):
            print(utility)
        optimizer.zero_grad()
        utility.backward()
        optimizer.step()
    t_tar = policy_dm(x_tensor)
    t_tar = t_tar.detach().cpu().numpy()
    print('DM Optimized Policy Value:', np.mean(outcome_model.GetOutcome(x, t_tar)))
    l3.append(np.mean(outcome_model.GetOutcome(x, t_tar)))

    # policy_dr = PolicyNet(p, 10, 3)
    # optimizer = torch.optim.Adam(policy_dr.parameters(), lr = 0.001)
    # x_tensor = torch.Tensor(x)
    # iter_dr = 8000
    # for it in range(iter_dr):
    #     t_tar = policy_dr(x_tensor)
    #     utility = -policy_eva.DRestimator(x, t, y, ps, t_tar, reg)
    #     if ((it + 1) % (iter_dr / 50) == 0):
    #         print(utility)
    #     optimizer.zero_grad()
    #     utility.backward()
    #     optimizer.step()
    # t_tar = policy_dr(x_tensor)
    # t_tar = t_tar.detach().cpu().numpy()
    # print('DR Optimized Policy Value:', np.mean(outcome_model.GetOutcome(x, t_tar)))
    # l4.append(np.mean(outcome_model.GetOutcome(x, t_tar)))

l1 = np.array(l1)
print('IPS', l1.mean(), l1.std())
l2 = np.array(l2)
print('SNIPS', l2.mean(), l2.std())
l3 = np.array(l3)
print('DM', l3.mean(), l3.std())
l4 = np.array(l4)
print('DR', l4.mean(), l4.std())
