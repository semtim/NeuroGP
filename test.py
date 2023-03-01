from scipy import stats
import pandas as pd
#from snad.load.curves import OSCCurve
import os
import neuroGP
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
#######################################################################
def data_gen(n=20):
    x_train = np.random.rand(n) * 150 - 50
    x_train.sort()
    y = x_train*2  #stats.norm.rvs(0, 5, size=n) #0.5*x_train 
    #X = np.vstack((X, X[:, ::-1]))
    #X, Y = torch.tensor(X).float(), torch.tensor(Y).float().view(-1, 1)
    return x_train, y

n = 20
x, y = data_gen(n)
#x = (x - np.mean(x))#/np.std(x)
#y_norm = y/np.max(y)
err = np.ones(len(x))*1e-3

n_epoch = 50 #50 for rbf with lr=2
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpr = neuroGP.NeuroGP(n_epoch=n_epoch, device='cpu', init_form='normal')

#hooks_data_history = neuroGP.register_model_hooks(gpr.kernel)

gpr.fit([x], [y], [err])

#neuroGP.plot_hooks_data(hooks_data_history)

fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
plt.plot(np.arange(1,n_epoch+1), np.array(gpr.ep_loss))


X = np.linspace(np.min(x), np.max(x), 100)
E = gpr.predict(x, y, err, X)
fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
plt.plot(X, E)
plt.scatter(x, y)


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

sk_gpr = GaussianProcessRegressor(kernel=RBF(), alpha=err**2).fit(x.reshape(-1,1), y)
plt.plot(X, sk_gpr.predict(X.reshape(-1,1)))
plt.scatter(x,y)

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

gpr = neuroGP.NeuroGP()
gpr.fit(x, y, err)


X = np.linspace(min(x[0]), max(x[0]), 380)
E = gpr.predict(x[0], y[0], err[0], X)
plt.plot(X, E)
plt.scatter(x[10], y[10])
