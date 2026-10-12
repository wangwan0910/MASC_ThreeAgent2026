"""Environments and multi-agent wrapper (no Ray cluster needed)."""
import numpy as np
import pytest

from masc.marl_env import ENV_IDS, ENV_SHORT_NAMES, SupplyChainMARLEnv, resolve_env_class

FIXED = {"factory": [0, 18, 0], "warehouse": [0, 18, 0], "retailer": [0, 18, 6]}


def rollout(env, policy=FIXED, seed=0):
    """Episode with a fixed policy; returns (sum of the team_reward infos, number of decisions, last terminated)."""
    obs, info = env.reset(seed=seed)
    total, n, done = info["factory"]["team_reward"], 0, False
    while not done:
        obs, rewards, terminated, truncated, info = env.step({a: np.array(v) for a, v in policy.items()})
        assert all(env.observation_spaces[a].contains(o) for a, o in obs.items())
        total += info["factory"]["team_reward"]
        n += 1
        done = terminated["__all__"] or truncated["__all__"]
    return total, n, terminated["__all__"]


@pytest.mark.parametrize("env_id", sorted(ENV_IDS))
def test_episode_runs_and_costs_add_up(env_id):
    env = SupplyChainMARLEnv({"env_class": env_id, "max_stockout_days": 0})
    total, n, terminated = rollout(env)
    assert env.env.day == 100 and not terminated and n > 0
    # one definition of the episode cost: team_reward infos == Cost column of the wrapped environment
    assert total == pytest.approx(float(np.sum(env.env.reward_history)))
    assert len(env.env.reward_history) == 100


def test_short_names_cover_all_ids():
    assert set(ENV_SHORT_NAMES) == set(ENV_IDS)


def test_legacy_module_names():
    assert resolve_env_class("AssembySto:Stocha_P1") is resolve_env_class("StoSupplyEnv-v3")
    assert resolve_env_class("AssembyBat:Batch_L10") is resolve_env_class("BatSupplyEnv-v1")


def test_stockout_termination():
    lazy = {"factory": [0, 0, 0], "warehouse": [0, 0, 0], "retailer": [0, 0, 0]}
    _, _, terminated = rollout(SupplyChainMARLEnv({"env_class": "StoSupplyEnv-v3"}), lazy)
    assert terminated                                             # default: 12 stockout days
    env = SupplyChainMARLEnv({"env_class": "StoSupplyEnv-v3", "max_stockout_days": 0})
    _, _, terminated = rollout(env, lazy)
    assert not terminated and env.env.day == 100


def test_seed_reproducible():
    a = SupplyChainMARLEnv({"env_class": "StoSupplyEnv-v3", "max_stockout_days": 0})
    b = SupplyChainMARLEnv({"env_class": "StoSupplyEnv-v3", "max_stockout_days": 0})
    assert rollout(a, seed=5)[0] == rollout(b, seed=5)[0]
    assert np.array_equal(a.env.demand_history, b.env.demand_history)


@pytest.mark.parametrize("mode,weight", [("team", 1.0), ("individual", 0.0), ("mixed", 0.5)])
def test_reward_modes(mode, weight):
    env = SupplyChainMARLEnv({"env_class": "BatSupplyEnv-v1", "reward_mode": mode, "reward_scale": 1.0})
    costs = [1.0, 2.0, 4.0]
    rewards = env._rewards(costs)
    for aid, c in zip(env.possible_agents, costs):
        assert rewards[aid] == pytest.approx(-((1 - weight) * c + weight * sum(costs)))
