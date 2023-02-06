import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from collections import defaultdict



class Sigma(nn.Module):
    def __init__(self):
        super(Sigma, self).__init__()
        self.sigma = torch.nn.Parameter(torch.ones(1))
        self.sigma.requires_grad = True

    def forward(self, x):
        return self.sigma**2 * x


class NeuroKernel(nn.Module):
    def __init__(self, act_fun=nn.ReLU(), init_form=None, device='cpu', sigma=1):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Linear(2, 128),
                        nn.Tanh(),
                        nn.Linear(128, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                                    )

        self.init_form = init_form
        self.device = device
        self.sigma_param = Sigma()
        if self.init_form is not None:
            self.init()


    def forward(self, x):
        """x - time-vector,
           K - covariance matrix"""

        K = torch.eye(x.shape[0]).to(self.device)
        x_batch = torch.tensor([0, 0]).to(self.device)
        for i, t_i in enumerate(x):
            for j, t_j in enumerate(x[i:]):
                x_batch = torch.vstack((x_batch, torch.tensor([t_i, t_j]).to(self.device)))

        K_list = self.layers(x_batch[1:].float())

        last_col_num = 0
        for i, t_i in enumerate(x):
            for j, t_j in enumerate(x[i:]):
                if i!=(i + j):
                    K[i, i+j] = K_list[j + last_col_num]
            last_col_num = j + 1
        K =  self.sigma_param(torch.matmul(K.t(), K))

        return K.double()

    def init(self):
        tanh_gain = torch.nn.init.calculate_gain("tanh")
        leaky_gain = torch.nn.init.calculate_gain("leaky_relu")
        sigmoid_gain = torch.nn.init.calculate_gain("sigmoid")
        torch.nn.init.xavier_normal_(self.layers[0].weight, gain=tanh_gain)
        torch.nn.init.xavier_normal_(self.layers[2].weight, gain=tanh_gain)
        torch.nn.init.xavier_normal_(self.layers[4].weight, gain=tanh_gain)




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
        #print( K.shape ,y.shape, err.shape)
        K_y = K + torch.matmul(I, noise)
        n = y.shape[0]
        logp = -0.5 * torch.matmul(torch.matmul(K_y.inverse(), y), y) - \
                    0.5 * K_y.logdet() - n / 2 * np.log(2 * np.pi)
        mod_logp = logp / n
        
        # AIC = 2*n - 2*(-0.5 * torch.matmul(torch.matmul(K_y.inverse(), y), y) - \
        #             0.5 * K_y.logdet() - n / 2 * np.log(2 * np.pi))

        return -mod_logp



class NeuroGP():
    def __init__(self, n_epoch=10, init_form=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel = NeuroKernel(init_form=init_form, device=self.device).train()
        self.kernel.to(self.device)
        self.loss = LogLikelihood(device=self.device)
        self.optimizer = torch.optim.Adam(self.kernel.parameters(), lr=0.01)  # Weight update
        self.n_epoch = n_epoch

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
        for epoch in range(self.n_epoch):
            ep_loss = 0
            self.kernel.train()
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
    def predict(self, x, y, err, x_appr=torch.tensor([]), return_sigma=False):
        x_obs = torch.tensor(x)
        if len(x_appr):
            x_appr = torch.tensor(x_appr)
        else:
            x_appr = x_obs
        X = torch.hstack((x_appr, x_obs))
        y_obs = torch.tensor(y)
        K = self.kernel(X).cpu()
        K_appr, K_obs = K[:len(x_appr), :len(x_appr)], K[-len(x_obs):, -len(x_obs):]
        K_b = K[len(x_appr):, :len(x_appr)]

        noise = torch.tensor(err)**2
        I = torch.ones(K_obs.shape).double()
        K_y = K_obs + torch.matmul(I, noise)
        E = torch.matmul(torch.matmul(K_b.t(), K_y.inverse()), y_obs)
        sigma = K_appr - torch.matmul(torch.matmul(K_b.t(), K_y.inverse()), K_b)

        return (E, sigma) if return_sigma else E




def get_forward_hook(history_dict, key):
    def forward_hook(self, input_, output):
        history_dict[key] = input_[0].cpu().detach().numpy().flatten()

    return forward_hook


def get_backward_hook(history_dict, key):
    def backward_hook(grad):  # for tensors
        history_dict[key] = grad.abs().cpu().detach().numpy().flatten()

    return backward_hook


def register_model_hooks(model):
    cur_ind = 0
    hooks_data_history = defaultdict(list)
    for child in model.layers.children():
        if isinstance(child, nn.Linear):
            cur_ind += 1
            forward_hook = get_forward_hook(hooks_data_history, f"activation_{cur_ind}")
            child.register_forward_hook(forward_hook)

            backward_hook = get_backward_hook(hooks_data_history, f"gradient_{cur_ind}")
            child.weight.register_hook(backward_hook)
    return hooks_data_history


def plot_hooks_data(hooks_data_history):
    keys = hooks_data_history.keys()
    n_layers = len(keys) // 2

    activation_names = [f"activation_{i + 1}" for i in range(1, n_layers)]
    activations_on_layers = [
        hooks_data_history[activation] for activation in activation_names
    ]

    gradient_names = [f"gradient_{i + 1}" for i in range(n_layers)]
    gradients_on_layers = [hooks_data_history[gradient] for gradient in gradient_names]

    for plot_name, values, labels in zip(
        ["activations", "gradients"],
        [activations_on_layers, gradients_on_layers],
        [activation_names, gradient_names],
    ):
        fig, ax = plt.subplots(1, len(labels), figsize=(14, 4), sharey="row") 
        for label_idx, label in enumerate(labels):
            ax[label_idx].boxplot(values[label_idx], labels=[label])
        plt.show()
