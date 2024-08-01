"""Modules required for Lnets.

Code modified from: https://github.com/cemanil/LNets.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter

class DenseLinear(nn.Module):
    """Linear dense lnet layer featuring input, output and bias customization.
    """
    def __init__(self):
        super(DenseLinear, self).__init__()

    def _set_network_parameters(self, in_features, out_features, bias=True, cuda=False):
        self.in_features = in_features
        self.out_features = out_features

        # Set weights and biases.
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        nn.init.orthogonal_(self.weight, gain=stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        raise NotImplementedError

    def project_weights(self):
        with torch.no_grad():
            projected_weights = project_on_l2_ball(self.weight.t(), bjorck_iter=20,
            bjorck_order=1, bjorck_beta=0.5, cuda=False).t()

            # Override the previous weights.
            self.weight.data.copy_(projected_weights)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class BjorckLinear(DenseLinear):
    def __init__(self, in_features=1, out_features=1, bias=True):
        super(BjorckLinear, self).__init__()
        self._set_network_parameters(in_features, out_features, bias, cuda=False)

    def forward(self, x):
        # Scale the values of the matrix to make sure the singular values are
        # less than or equal to 1.
        scaling = get_safe_bjorck_scaling(self.weight, cuda=False) # or 1.0

        ortho_w = bjorck_orthonormalize(self.weight.t() / scaling).t()
        return F.linear(x, ortho_w, self.bias)


def bjorck_orthonormalize(w, beta=0.5, iters=20, order=1):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix"
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """

    if order == 1:
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w = (1 + beta) * w - beta * w.mm(w_t_w)

    elif order == 2:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w = (+ (15 / 8) * w
                 - (5 / 4) * w.mm(w_t_w)
                 + (3 / 8) * w.mm(w_t_w_w_t_w))

    elif order == 3:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w)

            w = (+ (35 / 16) * w
                 - (35 / 16) * w.mm(w_t_w)
                 + (21 / 16) * w.mm(w_t_w_w_t_w)
                 - (5 / 16) * w.mm(w_t_w_w_t_w_w_t_w))

    elif order == 4:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)

        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w)
            w_t_w_w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w_w_t_w)

            w = (+ (315 / 128) * w
                 - (105 / 32) * w.mm(w_t_w)
                 + (189 / 64) * w.mm(w_t_w_w_t_w)
                 - (45 / 32) * w.mm(w_t_w_w_t_w_w_t_w)
                 + (35 / 128) * w.mm(w_t_w_w_t_w_w_t_w_w_t_w))

    else:
        print("The requested order for orthonormalization is not supported. ")
        exit(-1)

    return w


def get_safe_bjorck_scaling(weight, cuda=False):
    bjorck_scaling = torch.tensor([np.sqrt(weight.shape[0] * weight.shape[1])]).float()
    bjorck_scaling = bjorck_scaling.cuda() if cuda else bjorck_scaling

    return bjorck_scaling


def project_on_l2_ball(weight, bjorck_iter, bjorck_order, bjorck_beta=0.5, cuda=False):
    with torch.no_grad():
        # Run Bjorck orthonormalization procedure to project the matrices on the orthonormal matrices manifold.
        ortho_weights = bjorck_orthonormalize(weight.t(),
                                              beta=bjorck_beta,
                                              iters=bjorck_iter,
                                              order=bjorck_order).t()

        return ortho_weights

