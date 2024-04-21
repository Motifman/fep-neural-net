import torch
import torch.nn as nn
import numpy as np
import torch.jit as jit
from typing import List
from torch.distributions import Bernoulli, kl_divergence


class leaky_rnn_cell(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, unimix=0.01):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.randn([input_size, hidden_size]))
        self.weight_hh = nn.Parameter(torch.randn([hidden_size, hidden_size]))
        self.alpha = alpha
        self.unimix = unimix

    def forward(self, x, state):
        input_term = torch.mm(x, self.weight_ih)
        hidden_term = torch.mm(state["spike"], self.weight_hh)
        prob = torch.sigmoid(
            self.alpha * state["prob"] + (1 - self.alpha) * (input_term + hidden_term)
        )
        uniform = torch.ones_like(prob) / 2
        prob = (1 - self.unimix) * prob + self.unimix * uniform
        pred = torch.sigmoid(torch.mm(prob, self.weight_hh.t()))
        spike = Bernoulli(prob).sample()
        spike = spike + (prob - prob.detach())
        out = {"prob": prob, "spike": spike, "pred": pred}
        state = {"prob": prob, "spike": spike}
        return out, state


class RNNLayer(nn.Module):
    def __init__(self, cell, *cell_args) -> None:
        super().__init__()
        self.cell = cell(*cell_args)

    def forward(self, inputs, state):
        # (T, B, I), (T, B, H)
        inputs = inputs.unbind(0)
        spikes = [state["spike"]]
        preds = []
        probs = []
        for t in range(len(inputs)):
            out, state = self.cell(inputs[t], state)
            spikes += [out["spike"]]
            preds += [out["pred"]]
            probs += [out["prob"]]
        out = {
            "spikes": torch.stack(spikes),
            "preds": torch.stack(preds),
            "probs": torch.stack(probs),
        }
        return out, state


class FepNet(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, mu) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_layer = RNNLayer(leaky_rnn_cell, input_size, hidden_size, alpha)
        self.w_ih = self.rnn_layer.cell.weight_ih
        self.w_hh = self.rnn_layer.cell.weight_hh
        self.eps = 1e-6
        self.mu = mu

    def forward(self, inputs):
        device = inputs.device
        _, B, _ = inputs.shape
        prob_0 = torch.zeros([B, self.hidden_size]).to(device)
        spike_0 = torch.zeros([B, self.hidden_size]).to(device)
        out, state = self.rnn_layer(inputs, {"prob": prob_0, "spike": spike_0})

        return out, state

    def free_energy_loss(self, x):
        # x:(T, B, I)
        # RNN dynamics
        out, _ = self(x)  # spikes:(T+1, B, H), preds:(T, B, H)

        # reconst term
        reconst_loss = self._reconst_term(out["spikes"][:-1], out["preds"])

        # kl term
        kl_loss = self._kl_term(out["probs"])
        return reconst_loss + kl_loss

    def _reconst_term(self, obs, preds):
        reconst = obs * torch.log(preds + self.eps) + (1 - obs) * torch.log(
            1 - preds + self.eps
        )
        return -reconst.sum((0, 2)).mean()  # (T, B, H)->(B)->()

    def _kl_term(self, probs):
        mu = torch.ones_like(probs) * self.mu
        # kl = probs * (self._log(probs) - self._log(mu)) + (1 - probs) * (
        #     self._log(1 - probs) - self._log(1 - mu)
        # )
        kl = kl_divergence(Bernoulli(probs), Bernoulli(mu))
        return kl.sum((0, 2)).mean()

    def _log(self, x):
        return torch.log(x + self.eps)


if __name__ == "__main__":
    B, T, I, H, alpha, mu = 1, 15, 10, 20, 0.3, 0.2
    model = FepNet(I, H, alpha, mu)
    inputs = torch.randn(T, B, I)
    outputs, _ = model(inputs)
    loss = model.free_energy_loss(inputs)
