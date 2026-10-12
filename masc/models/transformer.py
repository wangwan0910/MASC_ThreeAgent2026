# -*- coding: utf-8 -*-
"""
Gated transformer policy for PPO (Ray RLlib new API stack), registered as model "transformer".

Modelled on the attention net example of RLlib (ray-2.6.1, rllib/examples/attention_net.py: PPO with the GTrXL of
Parisotto et al. 2019, "Stabilizing Transformers for Reinforcement Learning", model option use_attention), which only
exists for the old API stack (ModelV2). This module brings the same architecture to the RLModule API used by masc:

    observation (memory + 1 steps) -> linear input projection + learned position embedding
        -> num_layers x [ LayerNorm -> multi-head attention -> GRU gate,  LayerNorm -> position-wise MLP -> GRU gate ]
        -> LayerNorm -> output of the current step -> policy head (logits) / value head

Memory: the module keeps the last `memory` observations of the episode as its state (STATE_IN / STATE_OUT), and
every step attends over a sliding window of these observations and the current one; memory slots from before the
start of the episode are masked. The window is computed the same way for one step (sampling) and for a sequence of
max_seq_len steps (training), so both give the same outputs. (The GTrXL of the old API stack keeps the hidden states
of every layer as memory instead of the observations.)

Settings (model_config) and the corresponding options of attention_net.py:
    memory       attention_memory_inference / attention_memory_training   past observations attended to
    num_layers   attention_num_transformer_units
    d_model      attention_dim
    num_heads    attention_num_heads
    head_dim     attention_head_dim
    ff_dim       attention_position_wise_mlp_dim
    gate_bias    attention_init_gru_gate_bias   (> 0: the gates start close to the identity, as in GTrXL)
    head_hiddens hidden layers of the policy and value heads (on top of the shared transformer)
    max_seq_len  max_seq_len   length of the training sequences
"""

import math

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import TorchCategorical, TorchMultiCategorical
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule

from . import register_model


class GRUGate(nn.Module):
    """GRU-type gating of GTrXL: out = (1 - z) * x + z * h, z close to 0 at the start (bias gate_bias)."""

    def __init__(self, d, gate_bias):
        super().__init__()
        self.w = nn.Linear(d, 3 * d, bias=False)      # r, z, h from the sublayer output y
        self.u = nn.Linear(d, 2 * d, bias=False)      # r, z from the residual stream x
        self.u_h = nn.Linear(d, d, bias=False)        # h from r * x
        self.gate_bias = gate_bias

    def forward(self, x, y):
        wr, wz, wh = self.w(y).chunk(3, dim=-1)
        ur, uz = self.u(x).chunk(2, dim=-1)
        r = torch.sigmoid(wr + ur)
        z = torch.sigmoid(wz + uz - self.gate_bias)
        h = torch.tanh(wh + self.u_h(r * x))
        return (1 - z) * x + z * h

class MultiHeadAttention(nn.Module):
    def __init__(self, d, num_heads, head_dim):
        super().__init__()
        self.h, self.dh = num_heads, head_dim
        self.qkv = nn.Linear(d, 3 * num_heads * head_dim, bias=False)
        self.out = nn.Linear(num_heads * head_dim, d)

    def forward(self, x, key_padding_mask):
        n, k, _ = x.shape
        q, kk, v = self.qkv(x).view(n, k, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)   # 3 x (n, h, k, dh)
        scores = q @ kk.transpose(-1, -2) / math.sqrt(self.dh)
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        return self.out((scores.softmax(-1) @ v).transpose(1, 2).reshape(n, k, self.h * self.dh))

class GatedTransformerLayer(nn.Module):
    def __init__(self, d, num_heads, head_dim, ff_dim, gate_bias):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, num_heads, head_dim)
        self.ff = nn.Sequential(nn.Linear(d, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d))
        self.gate1, self.gate2 = GRUGate(d, gate_bias), GRUGate(d, gate_bias)

    def forward(self, x, key_padding_mask):
        x = self.gate1(x, torch.relu(self.attn(self.ln1(x), key_padding_mask)))
        return self.gate2(x, torch.relu(self.ff(self.ln2(x))))

def mlp(sizes, out):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.Tanh()]
    return nn.Sequential(*layers, nn.Linear(sizes[-1], out))

class TransformerPPOTorchRLModule(TorchRLModule, ValueFunctionAPI):
    """Gated transformer over the last `memory` observations + policy / value heads (see the module docstring).
    Inputs: OBS (B, T, obs_dim) and STATE_IN {"memory": (B, memory, obs_dim), "valid": (B, memory)};
    outputs have the time axis T (T = 1 when sampling)."""

    def setup(self):
        c = self.model_config
        self.memory = int(c["memory"])
        self.window = self.memory + 1
        obs_dim = int(np.prod(self.observation_space.shape))
        d = int(c["d_model"])
        self.input_proj = nn.Linear(obs_dim, d)
        self.position = nn.Parameter(torch.randn(self.window, d) * 0.02)
        self.layers = nn.ModuleList(GatedTransformerLayer(d, int(c["num_heads"]), int(c["head_dim"]),
                                                          int(c["ff_dim"]), float(c["gate_bias"]))
                                    for _ in range(int(c["num_layers"])))
        self.final_ln = nn.LayerNorm(d)
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            n_logits = int(np.sum(self.action_space.nvec))
            self.action_dist_cls = TorchMultiCategorical.get_partial_dist_cls(
                space=self.action_space, input_lens=[int(n) for n in self.action_space.nvec])
        elif isinstance(self.action_space, gym.spaces.Discrete):
            n_logits, self.action_dist_cls = int(self.action_space.n), TorchCategorical
        else:
            raise ValueError(f"transformer: action space {self.action_space} not supported "
                             "(Discrete / MultiDiscrete)")
        hiddens = [d] + list(c["head_hiddens"])
        self.pi_head = mlp(hiddens, n_logits)
        self.vf_head = mlp(hiddens, 1)
        self._obs_dim = obs_dim

    def get_initial_state(self):
        return {"memory": np.zeros((self.memory, self._obs_dim), dtype=np.float32),
                "valid": np.zeros((self.memory,), dtype=np.float32)}

    def _embeddings(self, batch):
        obs = batch[Columns.OBS].float()
        obs = obs.reshape(obs.shape[0], obs.shape[1], -1)                      # (B, T, obs_dim)
        state = batch[Columns.STATE_IN]
        b, t = obs.shape[:2]
        seq = torch.cat([state["memory"].float(), obs], dim=1)                 # (B, memory + T, obs_dim)
        valid = torch.cat([state["valid"].float(), torch.ones(b, t, device=obs.device)], dim=1)
        # window of the step t: the `window` observations ending at t
        windows = seq.unfold(1, self.window, 1).permute(0, 1, 3, 2)           # (B, T, window, obs_dim)
        masks = valid.unfold(1, self.window, 1) < 0.5                          # (B, T, window), True: masked
        x = self.input_proj(windows.reshape(b * t, self.window, -1)) + self.position
        mask = masks.reshape(b * t, self.window)
        for layer in self.layers:
            x = layer(x, mask)
        embeddings = self.final_ln(x[:, -1]).reshape(b, t, -1)
        state_out = {"memory": seq[:, -self.memory:], "valid": valid[:, -self.memory:]}
        return embeddings, state_out

    def _forward(self, batch, **kwargs):
        embeddings, state_out = self._embeddings(batch)
        return {Columns.ACTION_DIST_INPUTS: self.pi_head(embeddings), Columns.STATE_OUT: state_out}

    def _forward_train(self, batch, **kwargs):
        embeddings, state_out = self._embeddings(batch)
        return {Columns.ACTION_DIST_INPUTS: self.pi_head(embeddings), Columns.STATE_OUT: state_out,
                Columns.EMBEDDINGS: embeddings}

    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings, _ = self._embeddings(batch)
        return self.vf_head(embeddings).squeeze(-1)


DEFAULTS = {"memory": 20, "num_layers": 1, "d_model": 64, "num_heads": 2, "head_dim": 32, "ff_dim": 64,
            "gate_bias": 2.0, "head_hiddens": (64,), "max_seq_len": 20}


@register_model("transformer", defaults=DEFAULTS)
def transformer_spec(config):
    """Gated transformer (GTrXL-style) over the last observations, shared by the policy and value heads."""
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    config = dict(config)
    config["head_hiddens"] = list(config["head_hiddens"])
    if config["memory"] < 1:
        raise ValueError("transformer: memory must be >= 1")
    return RLModuleSpec(module_class=TransformerPPOTorchRLModule, model_config=config)
