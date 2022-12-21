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
                        nn.Linear(2, 128),
                        nn.Tanh(),
                        nn.Linear(128, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                                    )

    def forward(self, x):
        return self.layers(x)

def rbf(dt, l=50, sigma=1):
    return sigma**2 * np.exp(-dt**2/l**2/2)

@torch.inference_mode()
def predict(x, model):
    return model(x)

n = 100
X = np.random.rand(n//2, 2) * 100
Y = rbf(X[:,0] - X[:,1])

X, Y = torch.tensor(X).float(), torch.tensor(Y).float().view(-1, 1)

model = rbfNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
ep_loss = []
for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, Y)
    loss.backward()
    optimizer.step()
    ep_loss.append( loss.item() )


fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
plt.plot(np.arange(100), ep_loss)
ax.set_xlabel('epoch')
ax.set_ylabel('MSE loss')

fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
plt.scatter(np.arange(n//2), pred.detach().numpy().reshape(-1), c='red')
plt.scatter(np.arange(n//2), Y, c='blue')

x_apr = np.random.rand(5, 2) * 100
y = rbf(x_apr[:,0] - x_apr[:,1])
x_apr, y = torch.tensor(x_apr).float(), torch.tensor(y).float().view(-1, 1)
y_apr = predict(x_apr, model)

criterion(y_apr, y)

fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
plt.scatter(x_apr[:,0] - x_apr[:,1], y_apr, c='red')
plt.scatter(x_apr[:,0] - x_apr[:,1], y, c='blue')
plt.legend(['rbf prediction', 'rbf'])
ax.set_xlabel('t1-t2')
ax.set_ylabel('RBF(t1-t2)')