from scipy import stats
import pandas as pd
from snad.load.curves import OSCCurve
import os
import neuroGP
import numpy as np
import matplotlib.pyplot as plt

#######################################################################

x = np.linspace(0, 10, 10)
y = 0.5*x #np.sin(x)
x_norm = (x - np.mean(x))/np.std(x)
y_norm = y/np.max(y)
err = np.ones(len(x))*1e-6
gpr = neuroGP.NeuroGP(init_form='normal')

hooks_data_history = neuroGP.register_model_hooks(gpr.kernel)

gpr.fit([x], [y], [err])

neuroGP.plot_hooks_data(hooks_data_history)

fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
plt.plot(np.arange(1,101), np.log10(np.array(gpr.ep_loss)))


X = np.linspace(0, 10, 100)
fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
E = gpr.predict(x, y, err, X)
plt.plot(X, E)
plt.plot(x, y)


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

sk_gpr = GaussianProcessRegressor(kernel=RBF(), alpha=err**2).fit(x.reshape(-1,1), y)
plt.plot(X, sk_gpr.predict(X.reshape(-1,1)))
plt.plot(x,y)

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
