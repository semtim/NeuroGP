import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from snad.load.curves import OSCCurve
import os



class NeuroKernel(nn.Module):
    def __init__(self, act_fun=nn.ReLU(), init_form=None, device='cpu'):
        super().__init__()
        self.layers = nn.Sequential(
                                    nn.Linear(2, 128),
                                    #nn.BatchNorm1d(512),
                                    nn.Sigmoid(),
                                    nn.Linear(128, 32),
                                    #nn.BatchNorm1d(64),
                                    #nn.Dropout(0.7),
                                    nn.ReLU(),
                                    nn.Linear(32, 1),
                                    #nn.ReLU(),
                                    #nn.Linear(16, 1),
                                    )

        self.init_form = init_form
        self.device = device
        if self.init_form is not None:
            self.init()


    def forward(self, x, x_appr=torch.tensor([])):
        """x - observed time-vector
            K - covariance matrix"""
        if not x_appr.numel():
            K = torch.eye(x.shape[0]).to(self.device)
            x_batch = torch.tensor([0, 0]).to(self.device)
            for i, t_i in enumerate(x):
                for j, t_j in enumerate(x[i:]):
                    x_batch = torch.vstack((x_batch, torch.tensor([t_i, t_j]).to(self.device)))
        
            K_list = self.layers(x_batch[1:].float())
            
            last_col_num = 0
            for i, t_i in enumerate(x):
                for j, t_j in enumerate(x[i:]):
                    K[i, i+j] = K_list[j + last_col_num]
                last_col_num = j + 1
            #K += K.t() - torch.diag(K)
            K = torch.matmul(K.t(), K)
        #ниже часть для построения ядра для predict, пока не доделал
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

    def init(self):
        gain = torch.nn.init.calculate_gain("sigmoid")
        for child in self.layers.children():
            if isinstance(child, nn.Linear):
                if self.init_form == "normal":
                    torch.nn.init.xavier_normal_(child.weight, gain=gain)
                    if child.bias is not None:
                        torch.nn.init.zeros_(child.bias)
                elif self.init_form == "uniform":
                    torch.nn.init.xavier_uniform_(child.weight, gain=gain)
                    if child.bias is not None:
                        torch.nn.init.zeros_(child.bias)
                elif self.init_form == "kaiming_normal_":
                    torch.nn.init.kaiming_normal_(child.weight, nonlinearity='sigmoid')
                    if child.bias is not None:
                        torch.nn.init.zeros_(child.bias)




class LogLikelihood(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def forward(self, K, y, err):
        """K - covariance matrix,
        y - observed data,
        err - y errors"""
        noise = err**2
        I = torch.ones(K.shape).double().to(self.device)
        K_y = K + torch.matmul(I, noise)
        n = y.shape[0]
        logp = -0.5 * torch.matmul(torch.matmul(K_y.inverse(), y), y) - \
                    0.5 * K_y.logdet() - n / 2 * np.log(2 * np.pi)
        mod_logp = logp / n
        
        # AIC = 2*n - 2*(-0.5 * torch.matmul(torch.matmul(K_y.inverse(), y), y) - \
        #             0.5 * K_y.logdet() - n / 2 * np.log(2 * np.pi))

        return -mod_logp



class NeuroGP():
    def __init__(self, init_form=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel = NeuroKernel(init_form=init_form, device=self.device).train()
        self.kernel.to(self.device)
        self.loss = LogLikelihood(device=self.device)
        self.optimizer = torch.optim.RMSprop(self.kernel.parameters(), alpha=0.9)  # Weight update
        

    def fit(self, x, y, err):
        """x - time-vectors list,
        y - obseved data list,
        err - errors list"""
        x_obs = x.copy()
        y_obs = y.copy()
        err_obs = err.copy()
        for i, sample in enumerate(x_obs):
            x_obs[i] = torch.tensor(sample).double().to(self.device)
            y_obs[i] = torch.tensor(y_obs[i]).double().to(self.device)
            err_obs[i] = torch.tensor(err_obs[i]).double().to(self.device)

        self.ep_loss = []
        for epoch in range(100):
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


#######################################################################
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

#######################################################################

x = np.linspace(0, 100, 150)
y = 3*x + 5 #np.sin(x)
err = np.ones(len(x))*0.01
gpr = NeuroGP(init_form='sigmoid')
gpr.fit([(x-np.mean(x))/np.std(x)], [y/np.max(y)], [err])

fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
plt.plot(np.arange(21,101), gpr.ep_loss[20:])