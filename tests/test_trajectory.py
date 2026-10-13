"""Trajectory CSV of the trained policies of every seed (masc.trajectory), with untrained modules as checkpoints."""
import csv
import os

import pytest

from masc.marl_env import SupplyChainMARLEnv, build_ppo_config
from masc.trajectory import write_trajectories

ENV_CONFIG = {"env_class": "BatSupplyEnv-v1", "max_stockout_days": 0}


def fake_checkpoint(path, model):
    """Checkpoint directory with the layout of an RLlib algorithm checkpoint (only the RLModules)."""
    config = build_ppo_config(ENV_CONFIG, model=model, seed=0)
    spec = config.rl_module_spec
    env = SupplyChainMARLEnv(ENV_CONFIG)
    for aid, s in spec.rl_module_specs.items():
        s.observation_space, s.action_space = env.observation_spaces[aid], env.action_spaces[aid]
    spec.build().save_to_path(os.path.join(path, "learner_group", "learner", "rl_module"))
    return str(path)


@pytest.mark.parametrize("model", ["mlp", "transformer"])
def test_one_episode_per_seed(tmp_path, model):
    trials = [{"seed": seed, "checkpoint": fake_checkpoint(tmp_path / f"ck{seed}", model)} for seed in (1, 0)]
    out = tmp_path / "trajectory.csv"
    summary = write_trajectories(out, trials, ENV_CONFIG)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 200 and [s[0] for s in summary] == [0, 1]           # sorted by seed, 100 days each
    assert rows[0]["seed"] == "0" and rows[0]["demand_seed"] == "0" and rows[0]["day"] == "1"
    assert rows[-1]["seed"] == "1" and rows[-1]["day"] == "100"
    for seed, _, days, cost in summary:
        assert days == 100
        assert sum(float(r["Cost"]) for r in rows if r["seed"] == str(seed)) == pytest.approx(-cost)
    with pytest.raises(FileExistsError):                                    # never overwrites
        write_trajectories(out, trials, ENV_CONFIG)
    same = write_trajectories(tmp_path / "same_demand.csv", trials, ENV_CONFIG, csv_seed=7)
    assert [s[1] for s in same] == [7, 7]
