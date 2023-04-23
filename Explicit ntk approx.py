## **`NTK Approximation `**


import os
import torch
from torch import nn


# Define the neural net with pytorch
class nn_arch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, actv_fn: str = "ReLU"):
        """
        Arbitrary activation function from the list below:
        [ReLU, Sigmoid, Tanh, LeakyReLU, etc]

        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.actv_fn = getattr(nn, actv_fn)()
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)

        # initializing the weigths
        torch.nn.init.normal_(self.layer1.weight, mean=0.0, std=1 / np.sqrt(input_size))
        torch.nn.init.normal_(self.layer2.weight, mean=0.0, std=1 / np.sqrt(hidden_size))

    def k(self, x: torch.Tensor, y: torch.Tensor, j):
        jx = j(x)
        jy = j(y)
        if len(jx.size()) == 1:
            jx = jx.reshape(1, -1)
        if len(jy.size()) == 1:
            jy = jy.reshape(1, -1)
        return torch.matmul(jx, jy.transpose(0, 1))

    def j1(self, x: torch.Tensor):
        return x * self.layer2(self.derivative(self.layer1(x) / np.sqrt(self.input_size), self.actv_fn)) / np.sqrt(
            self.input_size * self.hidden_size)

    def j2(self, x):
        return self.actv_fn(self.layer1(x) / np.sqrt(self.input_size)) / np.sqrt(self.hidden_size)

    def kernel_calc(self, x: torch.Tensor, y: torch.Tensor):
        return self.k(x=x, y=y, j=self.j1) + self.k(x=x, y=y, j=self.j2)

    def forward(self, data: torch.Tensor):
        output = torch.flatten(data, 1)
        output = self.actv_fn(self.layer1(output))
        output = self.layer2(output)
        return output

    @staticmethod
    def derivative(x: torch.Tensor, f):
        p = torch.tensor(x, requires_grad=True)
        y_p = f(p)
        y_p.backward(torch.ones_like(p))
        return p.grad


torch_X_train = torch.from_numpy(X_train).to(torch.float32)
torch_X_test = torch.from_numpy(X_test).to(torch.float32)
torch_y_train = torch.from_numpy(y_train).to(torch.float32)

# Approximating the NTK
y_pred = []
widths = np.arange(10, 100000, 1000)
for width in widths:
    neural_net = nn_arch(input_size=torch_X_train.shape[1], hidden_size=width)
    output = neural_net(torch_X_train)
    k_train = neural_net.kernel_calc(torch_X_train, torch_X_train)
    k_test = neural_net.kernel_calc(torch_X_train, torch_X_test)
    pred = output + torch.matmul((torch_y_train - output).transpose(0, 1),
                                 torch.matmul(torch.inverse(k_train), k_test)).transpose(0, 1)
    y_pred.append(pred.detach().numpy())

timed_returns = np.array(y_pred).T * y_test.reshape(-1, 1)
sharpe_ratios = sharpe_ratio(timed_returns)

sr_plot(predictions=np.squeeze(y_pred).transpose(), returns=y_test)