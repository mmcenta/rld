import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4, bias=True, **kwargs):
        super(NoisyLinear, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class QNetwork(nn.Module):
    def __init__(self, obs_size, act_size, layers=[], dueling=False,
        noisy=False, **kwargs):
        super(QNetwork, self).__init__(**kwargs)

        self.obs_size = obs_size
        self.act_size = act_size
        self.dueling = dueling
        self.noisy = noisy

        linear = NoisyLinear if noisy else nn.Linear

        in_size = obs_size
        self.layers = nn.ModuleList([])
        for l in layers:
            self.layers.append(linear(in_size, l))
            in_size = l

        if not dueling:
            self.to_Q = linear(in_size, act_size) # state-action values
        else:
            self.to_V = linear(in_size, 1) # state value
            self.to_A = linear(in_size, act_size) # advantages

    def setcuda(self, device):
        self.cuda(device=device)

    def reset_noise(self):
        assert self.noisy
        for i in range(len(self.layers)):
            self.layers[i].reset_noise()
        if not self.dueling:
            self.to_Q.reset_noise()
        else:
            self.to_V.reset_noise()
            self.to_A.reset_noise()

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.tanh(x)
            x = self.layers[i](x)

        if not self.dueling:
            return self.to_Q(torch.tanh(x))

        values = self.to_V(torch.tanh(x))
        advantages = self.to_A(torch.tanh(x))
        adv_mean = torch.mean(advantages, dim=-1, keepdim=True)
        advantages -= adv_mean
        return values + advantages