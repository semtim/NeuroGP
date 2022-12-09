import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from snad.load.curves import OSCCurve
import os



class NeuroKernel(nn.Module):
    def __init__(self, act_fun=nn.ReLU()):
        super().__init__()
        self.fc1 = nn.Linear(2, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)
        self.activation = act_fun


    def forward(self, x, x_appr=torch.tensor([])):
        """x - observed time-vector
            K - covariance matrix"""
        if not x_appr.numel():
            K = torch.eye(x.shape[0])
            for i, t_i in enumerate(x):
                for j, t_j in enumerate(x[i:]):
                    current_x = torch.tensor([[t_i, t_j]]).float()
                    current_x = self.fc1(current_x)
                    current_x = self.activation(current_x)
                    current_x = self.fc2(current_x)
                    current_x = self.activation(current_x)
                    current_x = self.fc3(current_x)
                    K[i, i+j] = current_x
            #K += K.t() - torch.diag(K)
            K = torch.matmul(K.t(), K)
        else:
            """x[0] - observed time-vector,
            x[1] - approximated time-vector"""
            x_obs = x
            K = torch.eye(x_obs.shape[0], x_appr.shape[0])
            for i, t_i in enumerate(x_obs):
                for j, t_j in enumerate(x_appr):
                    current_x = torch.tensor([t_i, t_j, (t_i - t_j)**2]).float()
                    current_x = self.fc1(current_x)
                    current_x = self.activation(current_x)
                    current_x = self.fc2(current_x)
                    current_x = self.activation(current_x)
                    current_x = self.fc3(current_x)
                    K[i, j] = current_x

        return K.double()



class LogLikelihood(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, K, y, err):
        """K - covariance matrix,
        y - observed data,
        err - y errors"""
        noise = err**2
        I = torch.ones(K.shape).double()
        K_y = K + torch.matmul(I, noise)
        n = y.shape[0]
        logp = -0.5 * torch.matmul(torch.matmul(K_y.inverse(), y), y) - \
                    0.5 * K_y.logdet() - n / 2 * np.log(2 * np.pi)
        mod_logp = logp / n
        
        # AIC = 2*n - 2*(-0.5 * torch.matmul(torch.matmul(K_y.inverse(), y), y) - \
        #             0.5 * K_y.logdet() - n / 2 * np.log(2 * np.pi))

        return -mod_logp



class NeuroGP():
    def __init__(self):
        self.kernel = NeuroKernel().train()
        self.loss = LogLikelihood()
        self.optimizer = torch.optim.SGD(self.kernel.parameters(), lr=0.001)  # Weight update

    def fit(self, x, y, err):
        """x - time-vectors list,
        y - obseved data list,
        err - errors list"""
        x_obs = x.copy()
        y_obs = y.copy()
        err_obs = err.copy()
        for i, sample in enumerate(x_obs):
            x_obs[i] = torch.tensor(sample).double()
            y_obs[i] = torch.tensor(y_obs[i]).double()
            err_obs[i] = torch.tensor(err_obs[i]).double()

        self.ep_loss = []
        for epoch in range(40):
            ep_loss = 0
            for i, sample in enumerate(x_obs):
                self.optimizer.zero_grad()
                K = self.kernel(sample)
                loss = self.loss(K, y_obs[i], err_obs[i])
                loss.backward()
                self.optimizer.step()
                ep_loss += loss.item()
            #print(loss)
            self.ep_loss.append(ep_loss / len(x_obs))


    @torch.inference_mode()
    def predict(self, x, y, err, x_appr, return_sigma=False):
        x_obs = torch.tensor(x)
        x_appr = torch.tensor(x_appr)
        y_obs = torch.tensor(y)
        K_obs = self.kernel(x_obs)
        noise = torch.tensor(err)**2
        I = torch.ones(K_obs.shape).double()
        K_y = K_obs + torch.matmul(I, noise)
        K_b = self.kernel(x_obs, x_appr=x_appr)
        K_appr = self.kernel(x_appr)
        E = torch.matmul(torch.matmul(K_b.t(), K_y.inverse()), y_obs)
        sigma = K_appr - torch.matmul(torch.matmul(K_b.t(), K_y.inverse()), K_b)

        return (E, sigma) if return_sigma else E



temp = os.path.abspath("second_cut.csv")
name = pd.read_csv(temp, sep=",")
name = pd.DataFrame(name)

sn = []
for i in range(len(name)):
    try:
        sn.append(OSCCurve.from_json(os.path.join('./sne', name['Name'][i] + '.json'), bands='r'))
        sn[-1] = sn[-1].filtered(with_upper_limits=False, with_inf_e_flux=False, sort='filtered')
        sn[-1] = sn[-1].binned(bin_width=1, discrete_time=True)
    except:
        continue

x, y = [], []
err = []
for i in range(len(sn)):
    if len(sn[i].X[:,1]) >= 20:
        y.append(sn[i].y)
        x.append(sn[i].X[:,1])  # - sn[i].X[:,1][np.argmax(y[-1])]
        err.append(sn[i].err)

gpr = NeuroGP()
gpr.fit(x, y, err)


X = np.linspace(min(x[0]), max(x[0]), 380)
E = gpr.predict(x[0], y[0], err[0], X)
plt.plot(X, E)
plt.scatter(x[10], y[10])

fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
plt.plot(np.arange(1,41), gpr.ep_loss)