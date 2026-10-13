# -*- coding: utf-8 -*-
"""
Trajectory CSV of the trained (converged) policies: one deterministic episode (T = 100 days) of the final policies of
every trial (random seed) of a run, in one file: 5 seeds -> 500 rows. Written by masc-train / masc-train-transformer
at the end of the training, and by the masc-trajectory command for an existing run:

    masc-trajectory tu_seeds_logs/Bat10/1791804852                  # -> <run root>/csv_results/..._trajectory.csv
    masc-trajectory tu2_seeds_logs/Sto1/1791808000 --lr 0.0001 --csv-seed 0 --out sto1.csv

Columns: seed (training seed of the trial), demand_seed (seed of the demand of the episode), day (1..T), then the
columns of the environment's save_to_csv: <agent>_<state field> (inventory, reorderpoint, pendingDelivery,
daysDelivery, stockout_cum, processingTimes, stockout_daily, capacity, stockout_total; state after the day),
<agent>_<action field> (serviceTime, orderToSupplier, reorderPoint; executed on the reorder days, 0 on the other
days), customer_demand, Cost (cost booked on the day, negative; its sum over an episode is the episode total cost,
same definition as team_return) and retailer_actionTriggerHistory (1: reorder day).
The demand of the episode of a trial is drawn with its own training seed (0 if it has none), or with --csv-seed for
all the trials.
"""

import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np

from .marl_env import ENV_SHORT_NAMES, SupplyChainMARLEnv

PLOTTED = ("converged", "completed")   # verdicts of the trials whose policies are converged / fully trained


def rollout_episode(checkpoint, env_config, seed=0):
    """One episode of the trained policies (deterministic actions, state carried for the transformer policies).
    Returns the wrapped single-planner environment, whose histories hold the whole episode."""
    import torch
    import tree
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
    modules = MultiRLModule.from_checkpoint(os.path.join(os.path.abspath(checkpoint), "learner_group", "learner",
                                                         "rl_module"))
    env = SupplyChainMARLEnv(env_config)
    obs, _ = env.reset(seed=seed)
    states = {aid: tree.map_structure(lambda x: torch.as_tensor(x)[None], modules[aid].get_initial_state())
              for aid in env.possible_agents if modules[aid].is_stateful()}
    done = False
    with torch.no_grad():
        while not done:
            actions = {}
            for aid in env.possible_agents:
                module, o = modules[aid], torch.from_numpy(obs[aid])[None]
                batch = {"obs": o[:, None], "state_in": states[aid]} if aid in states else {"obs": o}
                out = module.forward_inference(batch)
                logits = out["action_dist_inputs"]
                if aid in states:
                    states[aid], logits = out["state_out"], logits[:, 0]   # remove the time dimension
                dist = module.get_inference_action_dist_cls().from_logits(logits)
                actions[aid] = dist.to_deterministic().sample()[0].numpy()
            obs, _, terminateds, truncateds, _ = env.step(actions)
            done = terminateds["__all__"] or truncateds["__all__"]
    return env.env


def write_trajectories(path, trials, env_config, csv_seed=None):
    """Trajectory CSV (see the module docstring) of the trials [{"checkpoint", "seed", ...}], sorted by seed.
    Returns [(seed, demand seed, days, episode total cost)]."""
    rows, summary, fieldnames = [], [], None
    for trial in sorted(trials, key=lambda t: (t.get("seed") is None, t.get("seed"))):
        seed = trial.get("seed")
        demand_seed = csv_seed if csv_seed is not None else (seed if seed is not None else 0)
        env = rollout_episode(trial["checkpoint"], env_config, seed=demand_seed)
        fieldnames = fieldnames or ["seed", "demand_seed", "day"] + env.csv_fieldnames()
        days = env.csv_rows()[1:]     # row 0 is the initial state, before the first day
        rows += [{"seed": "" if seed is None else seed, "demand_seed": demand_seed, "day": day, **row}
                 for day, row in enumerate(days, start=1)]
        summary.append((seed, demand_seed, len(days), -float(np.sum(env.reward_history))))
    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def unique_path(path):
    """path, or path with _2, _3 ... before the extension if it exists already (never overwrite an older file)."""
    root, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(path):
        k += 1
        path = f"{root}_{k}{ext}"
    return path


def demand_tag(env_config):
    """Demand setting of the environment for the file names, e.g. mu2-std1 (stochastic), mu2-std0.1-batch10 (batch)."""
    env = SupplyChainMARLEnv(env_config).env
    tag = f"mu{env.muDemand:g}-std{env.stdDemand:g}"
    return tag + (f"-batch{env.batch_size}" if hasattr(env, "batch_size") else "")


def print_summary(path, summary):
    costs = [c for *_, c in summary]
    print(f"csv: {path} ({len(summary)} seed(s) x {summary[0][2]} days, episode total cost "
          f"{np.mean(costs):.1f} +- {np.std(costs):.1f}: " +
          ", ".join(f"seed {s}: {c:.0f}" for s, _, _, c in summary) + ")")


# ----------------------------------------------------------------------------------------------------------------
# masc-trajectory
# ----------------------------------------------------------------------------------------------------------------

def run_trials(run_dir, lr=None):
    """Trials of a run (<dir>_logs/<env>/<time>/training_status.json) whose policies are converged / fully trained,
    of the learning rate `lr` (default: the learning rate with the best mean final return). Returns (trials, lr)."""
    path = os.path.join(run_dir, "training_status.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: the training of this run has not finished")
    with open(path) as f:
        status = [s for s in json.load(f) if s.get("status") in PLOTTED and s.get("checkpoint")]
    if not status:
        raise SystemExit(f"no converged / completed trial in {path}")
    lrs = sorted({s["lr"] for s in status})
    if lr is None:
        lr = max(lrs, key=lambda v: np.mean([s["episode_return_mean"] for s in status if s["lr"] == v]))
    trials = [s for s in status if math.isclose(s["lr"], lr)]
    if not trials:
        raise SystemExit(f"no converged / completed trial with lr={lr} (learning rates: {lrs})")
    return trials, lr


def trial_params(checkpoint):
    with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "params.json")) as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="masc-trajectory", description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("run_dir", help="log directory of a run, <dir>_logs/<env>/<time> (with training_status.json)")
    parser.add_argument("--lr", type=float, default=None, help="trials of this learning rate (default: the best)")
    parser.add_argument("--csv-seed", type=int, default=None,
                        help="demand seed of all the episodes (default: the training seed of every trial)")
    parser.add_argument("--out", default=None, help="CSV file (default: <run root>/csv_results/<name>_trajectory.csv)")
    args = parser.parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    trials, lr = run_trials(run_dir, args.lr)
    env_config = trial_params(trials[0]["checkpoint"])["env_config"]
    env_id = env_config["env_class"]
    experiment = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(trials[0]["checkpoint"]))))
    algo = "TPPO" if experiment.startswith(("TPPO", "TransformerPPO")) else "PPO"
    out = args.out
    if out is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(run_dir)))    # <root>/<dir>_logs/<env>/<time>
        os.makedirs(os.path.join(root, "csv_results"), exist_ok=True)
        seeds = f"_{len(trials)}seeds" if len(trials) > 1 else ""
        net = "gtrxl" if algo == "TPPO" else "mlp64x64"
        steps_m = f"{max(t['steps'] for t in trials) / 1e6:.2f}".rstrip("0").rstrip(".")
        out = os.path.join(root, "csv_results", f"{algo}_{ENV_SHORT_NAMES.get(env_id, env_id)}_{demand_tag(env_config)}"
                                                f"_lr{lr:g}_{net}_{steps_m}Msteps{seeds}"
                                                f"_{datetime.datetime.now():%Y%m%d-%H%M%S}_trajectory.csv")
    out = unique_path(out)
    print_summary(out, write_trajectories(out, trials, env_config, args.csv_seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
