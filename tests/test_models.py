"""Pluggable models (masc.models) and the transformer RLModule (no Ray cluster needed)."""
import pickle

import numpy as np
import pytest
import torch

from masc.marl_env import SupplyChainMARLEnv, build_ppo_config
from masc.models import build_rl_module_spec, model_defaults, model_names, register_model

ENV = SupplyChainMARLEnv({"env_class": "BatSupplyEnv-v1", "max_stockout_days": 0})


def build(name, config=None, agent="retailer"):
    spec = build_rl_module_spec(name, config)
    spec.observation_space = ENV.observation_spaces[agent]
    spec.action_space = ENV.action_spaces[agent]
    return spec.build()


def test_registry():
    assert {"mlp", "transformer"} <= set(model_names())
    with pytest.raises(ValueError):
        build_rl_module_spec("transformer", {"no_such_setting": 1})
    with pytest.raises(ValueError):
        build_rl_module_spec("nope")
    with pytest.raises(ValueError):
        register_model("mlp")(lambda c: None)                       # names are unique
    assert build_rl_module_spec("transformer", {"memory": None}).model_config["memory"] == \
        model_defaults("transformer")["memory"]                     # None keeps the default


def batch(obs, state):
    return {"obs": torch.as_tensor(obs, dtype=torch.float32), "state_in": state}


def initial_state(module, b=1):
    return {k: torch.as_tensor(v)[None].repeat(b, *([1] * v.ndim)) for k, v in module.get_initial_state().items()}


@pytest.mark.parametrize("config", [{}, {"memory": 3, "num_layers": 2, "num_heads": 1, "head_dim": 16}])
def test_transformer_step_by_step_equals_sequence(config):
    torch.manual_seed(0)
    module = build("transformer", config)
    module.eval()
    obs = np.random.RandomState(0).rand(1, 12, ENV.observation_spaces["retailer"].shape[0]).astype(np.float32)
    with torch.no_grad():
        # sampling: one step at a time, the state carried from step to step
        state, step_logits = initial_state(module), []
        for t in range(obs.shape[1]):
            out = module.forward_inference(batch(obs[:, t:t + 1], state))
            step_logits.append(out["action_dist_inputs"][:, 0])
            state = out["state_out"]
        step_logits = torch.stack(step_logits, dim=1)
        # training: the whole sequence at once, and two chunks with the state of the first one
        seq = module.forward_train(batch(obs, initial_state(module)))
        first = module.forward_train(batch(obs[:, :5], initial_state(module)))
        second = module.forward_train(batch(obs[:, 5:], first["state_out"]))
    assert seq["action_dist_inputs"].shape == (1, 12, int(np.sum(ENV.action_spaces["retailer"].nvec)))
    assert torch.allclose(seq["action_dist_inputs"], step_logits, atol=1e-5)
    assert torch.allclose(torch.cat([first["action_dist_inputs"], second["action_dist_inputs"]], 1), step_logits,
                          atol=1e-5)
    values = module.compute_values(batch(obs, initial_state(module)))
    assert values.shape == (1, 12)


def test_transformer_actions_and_memory_mask():
    module = build("transformer", {"memory": 4}, agent="factory")
    out = module.forward_exploration(batch(np.zeros((3, 1, 9), np.float32), initial_state(module, 3)))
    dist = module.get_exploration_action_dist_cls().from_logits(out["action_dist_inputs"][:, 0])
    actions = dist.sample()
    assert actions.shape == (3, 3) and all(ENV.action_spaces["factory"].contains(a.numpy()) for a in actions)
    assert torch.isfinite(dist.logp(actions)).all()
    # memory: the last 4 observations, the slots before the start of the episode are invalid
    assert out["state_out"]["memory"].shape == (3, 4, 9)
    assert out["state_out"]["valid"][0].tolist() == [0, 0, 0, 1]
    # a change of an observation older than the memory does not change the output
    torch.manual_seed(1)
    module = build("transformer", {"memory": 2})
    obs = torch.rand(1, 6, 9)
    other = obs.clone()
    other[:, 0] += 1.0
    with torch.no_grad():
        a = module.forward_train(batch(obs, initial_state(module)))["action_dist_inputs"]
        b = module.forward_train(batch(other, initial_state(module)))["action_dist_inputs"]
    assert not torch.allclose(a[:, 2], b[:, 2]) and torch.allclose(a[:, 3:], b[:, 3:])


def test_ppo_config_and_pickle():
    config = build_ppo_config({"env_class": "BatSupplyEnv-v1"}, model="transformer", model_config={"memory": 8})
    spec = config.rl_module_spec.rl_module_specs["factory"]
    assert spec.model_config["memory"] == 8 and spec.model_config["max_seq_len"] == 20
    pickle.loads(pickle.dumps(spec.module_class))                   # importable class (Ray workers, checkpoints)
    # default (no model): the RLlib default modules, as before the registry
    mlp = build_ppo_config({"env_class": "BatSupplyEnv-v1"}).rl_module_spec.rl_module_specs["factory"]
    assert mlp.model_config.fcnet_hiddens == [64, 64] and not mlp.model_config.use_lstm


def test_cli_parser():
    from masc import main_transformerPPO as tppo
    args = tppo.build_parser().parse_args(["--env", "BatSupplyEnv-v1", "--memory", "30", "--head-hiddens", "32", "32"])
    config = tppo.model_config_from_args(args)
    assert config["memory"] == 30 and config["head_hiddens"] == (32, 32) and args.dir == "tu2"
    assert build_rl_module_spec("transformer", config).model_config["memory"] == 30
