import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats


class rbfNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Linear(2, 50),
                        nn.Tanh(),
                        nn.Linear(50, 15),
                        nn.ReLU(),
                        nn.Linear(15, 1),
                                    )
        # self.init()

    def forward(self, x):
        return self.layers(x)
    
    # def init(self):
    #     tanh_gain = torch.nn.init.calculate_gain("tanh")
    #     relu_gain = torch.nn.init.calculate_gain("relu")
    #     sigmoid_gain = torch.nn.init.calculate_gain("sigmoid")
    #     torch.nn.init.xavier_normal_(self.layers[0].weight, gain=tanh_gain)
    #     torch.nn.init.xavier_normal_(self.layers[2].weight, gain=relu_gain)
    #     torch.nn.init.xavier_normal_(self.layers[4].weight, gain=relu_gain)

    

def rbf(dt, l=50, sigma=1):
    return sigma**2 * np.exp(-dt**2/l**2/2)

@torch.inference_mode()
def predict(x, model):
    model.cpu()
    res = model(x)
    model.to(dev)
    return res

n = 1000
def data_gen(n=20):
    x_train = np.random.rand(n) * 150 - 50
    x_train.sort()
    X = np.zeros(2)
    for i, x in enumerate(x_train):
        X = np.vstack((X, np.hstack((np.ones(x_train[i:].reshape(-1,1).shape)*x,
                                     x_train[i:].reshape(-1,1)))))
    X = X[1:]
    Y = rbf(X[:,0] - X[:,1])
    #X = np.vstack((X, X[:, ::-1]))
    X, Y = torch.tensor(X).float(), torch.tensor(Y).float().view(-1, 1)
    return X, Y


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, Y = data_gen(n)
X, Y = X.to(dev), Y.to(dev)
model = rbfNet().to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
ep_loss = []
n_epoch = 500
for epoch in range(n_epoch):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, Y)
    loss.backward()
    optimizer.step()
    ep_loss.append( loss.item() )


fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
plt.plot(np.arange(n_epoch), ep_loss)
ax.set_xlabel('epoch')
ax.set_ylabel('MSE loss')

#fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
#plt.scatter(np.arange(n//2), pred.detach().numpy().reshape(-1), c='red')
#plt.scatter(np.arange(n//2), Y, c='blue')

x_apr, y = data_gen(30)
y_apr = predict(x_apr, model)

criterion(y_apr, y)

fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
plt.scatter(x_apr[:,0] - x_apr[:,1], y_apr, c='red')
plt.scatter(x_apr[:,0] - x_apr[:,1], y, c='blue')
plt.legend(['rbf prediction', 'rbf'])
ax.set_xlabel('t1-t2')
ax.set_ylabel('RBF(t1-t2)')