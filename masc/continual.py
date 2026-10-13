# -*- coding: utf-8 -*-
"""
Continual learning with multi-agent PPO (Ray RLlib) between the supply chain environments (masc-continual).
Every phase starts from the policies of the previous phase; the sequence of a task is repeated --rounds times.

Tasks (--task), default number of rounds and directory of the models (under --out-dir):
    0-01-1     : std 0 -> 0.1 -> 1           3 rounds   PPOcon-sto_models/
    1-01-0     : std 1 -> 0.1 -> 0           3 rounds   PPOcon-sto_models/
    3-7-10     : batch size 3 -> 7 -> 10     3 rounds   PPOcon-bat_models/
    10-7-3     : batch size 10 -> 7 -> 3     3 rounds   PPOcon-bat_models/
    bat10-sto1 : batch 10 -> std 1           5 rounds   PPOconbat-sto_models/
    sto1-bat10 : std 1 -> batch 10           5 rounds   PPOconbat-sto_models/

    masc-continual --task 10-7-3                                   # lr 0.0003
    masc-continual --task 3-7-10 --rounds 1
    masc-continual --task sto1-bat10 --lr 0.001 0.0003 0.0001      # grid search, one lr after another

The training curve (mean episode return per iteration, cumulated environment steps) is written to TensorBoard,
<models_dir>/tb/, one scalar per phase, and plotted at the end of the task (one figure per learning rate, in the
style of assets/dcontinustoPPO2_2.png): <out-dir>/PPOcon_<task>_r<rounds>_lr<lr>_<date-time>.png (a copy stays in
<models_dir>); a new run never overwrites an older figure. The test reward of every phase is printed (no CSV file
is written by the continual learning for now).

The reward curve is checked during the training (ContinualMonitor); the task (of this learning rate) stops early,
skipping the remaining phases, when the trend stays bad:
    diverged   : non-finite loss or episode return -> stopped at once
    collapsed  : in any phase, the mean return of the last --monitor-window iterations stays more than
                 --collapse-tol below the best window of the phase for --collapse-patience iterations
    no_learning: in the first phase (trained from scratch), after min(--no-learning-steps, --timesteps-per-phase)
                 steps the best window is not --min-gain better than the first window
(the later phases start from trained policies, so not improving there is not a bad trend.)
A stopped task still plots its curve, as PPOcon_..._STOPPED-<reason>.png; the verdict is in
<models_dir>/lr=<lr>/continual_status.json, and the script exits with code 1 if every learning rate was stopped.
"""

import argparse
import datetime
import json
import math
import os
import shutil
import sys

import numpy as np
from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_LEARNER_GROUP, COMPONENT_RL_MODULE

from .marl_env import ENV_SHORT_NAMES, build_ppo_config, evaluate, init_ray
from .plotting import plot_continual


OBS_ROWS = 9
LRS = [3e-4]
TIMESTEPS_PER_PHASE = 200000
NUM_ENV_RUNNERS = 1
NUM_GPUS = 0

# continual learning tasks: order of the environments (see marl_env.ENV_IDS), default number of rounds (the order is
# repeated this many times) and directory of the models
# StoSupplyEnv-v1 = std 0 (demand always 2), StoSupplyEnv-v2 = std 0.1, StoSupplyEnv-v3 = std 1
# BatSupplyEnv-v1 = batch 10, BatSupplyEnv-v2 = batch 7, BatSupplyEnv-v3 = batch 3
TASKS = {
    "0-01-1": ["StoSupplyEnv-v1", "StoSupplyEnv-v2", "StoSupplyEnv-v3"],   # std 0 -> 0.1 -> 1
    "1-01-0": ["StoSupplyEnv-v3", "StoSupplyEnv-v2", "StoSupplyEnv-v1"],   # std 1 -> 0.1 -> 0
    "3-7-10": ["BatSupplyEnv-v3", "BatSupplyEnv-v2", "BatSupplyEnv-v1"],   # batch 3 -> 7 -> 10
    "10-7-3": ["BatSupplyEnv-v1", "BatSupplyEnv-v2", "BatSupplyEnv-v3"],   # batch 10 -> 7 -> 3
    "bat10-sto1": ["BatSupplyEnv-v1", "StoSupplyEnv-v3"],                  # batch 10 -> std 1
    "sto1-bat10": ["StoSupplyEnv-v3", "BatSupplyEnv-v1"],                  # std 1 -> batch 10
}
ROUNDS = {"0-01-1": 3, "1-01-0": 3, "3-7-10": 3, "10-7-3": 3, "bat10-sto1": 5, "sto1-bat10": 5}
MODELS_DIRS = {"0-01-1": "PPOcon-sto_models", "1-01-0": "PPOcon-sto_models",
               "3-7-10": "PPOcon-bat_models", "10-7-3": "PPOcon-bat_models",
               "bat10-sto1": "PPOconbat-sto_models", "sto1-bat10": "PPOconbat-sto_models"}
RL_MODULE_PATH = os.path.join(COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER, COMPONENT_RL_MODULE)


def make_phases(task, rounds):
    """[(phase name, env id)], e.g. [("round1_Sto0", "StoSupplyEnv-v1"), ...]"""
    return [(f"round{r}_{ENV_SHORT_NAMES[env_id]}", env_id) for r in range(1, rounds + 1) for env_id in TASKS[task]]


class ContinualMonitor:
    """Checks the reward curve of a continual learning task, phase by phase (see the module docstring).
    check() returns None while the trend is fine, else {"status", "phase", "steps", "reason"}."""

    def __init__(self, window=10, collapse_tol=0.2, collapse_patience=20, min_gain=0.05, no_learning_steps=400000):
        self.window, self.collapse_tol, self.collapse_patience = window, collapse_tol, collapse_patience
        self.min_gain, self.no_learning_steps = min_gain, no_learning_steps

    def start_phase(self, phase):
        self.phase, self.history, self.best, self.collapsed = phase, [], -np.inf, 0

    def _verdict(self, status, steps, reason):
        return {"status": status, "phase": self.phase, "steps": int(steps), "reason": reason}

    def check(self, result, steps, phase_length):
        ret = result["env_runners"].get("episode_return_mean")
        bad = [f"{m}/{k}" for m, metrics in (result.get("learners") or {}).items() if isinstance(metrics, dict)
               for k, v in metrics.items() if "loss" in k and isinstance(v, (int, float)) and not math.isfinite(v)]
        if bad:
            return self._verdict("diverged", steps, "non-finite loss " + ", ".join(bad))
        if ret is not None and math.isinf(ret):
            return self._verdict("diverged", steps, f"episode return {ret}")
        if ret is None or math.isnan(ret):   # no finished episode yet
            return None

        h, w = self.history, self.window
        h.append(ret)
        if len(h) < w:
            return None
        last, first = np.mean(h[-w:]), np.mean(h[:w])
        self.best = max(self.best, last)
        scale = max(abs(self.best), 1e-8)
        self.collapsed = self.collapsed + 1 if last < self.best - self.collapse_tol * scale else 0
        if self.collapsed >= self.collapse_patience:
            return self._verdict("collapsed", steps, f"return {last:.1f} stayed more than {self.collapse_tol:.0%} "
                                 f"below the best {self.best:.1f} of the phase for {self.collapse_patience} iterations")
        if self.phase == 1 and steps >= min(self.no_learning_steps, phase_length):
            gain = (self.best - first) / max(abs(first), 1e-8)
            if gain < self.min_gain:
                return self._verdict("no_learning", steps, f"best {self.best:.1f} only {gain:+.1%} better than the "
                                     f"start {first:.1f} after {steps} steps (from scratch)")
        return None


def figure_name(task, rounds, timesteps_per_phase, lr, stamp, seed=None, n_seeds=None, suffix=""):
    """File name of a continual learning figure, e.g.
    PPOcon_Sto0-Sto01-Sto1_3rounds_1Msteps-per-phase_lr0.0001_5seeds_<stamp>.png (n_seeds: figure of several seeds)."""
    envs = "-".join(ENV_SHORT_NAMES[e] for e in TASKS[task])
    seeds = f"_{n_seeds}seeds" if n_seeds else (f"_seed{seed}" if seed is not None else "")
    return (f"PPOcon_{envs}_{rounds}rounds_{timesteps_per_phase / 1e6:g}Msteps-per-phase_lr{lr:g}{seeds}"
            f"_{stamp}{suffix}.png")


def figure_title(task, rounds, timesteps_per_phase, lr, seed=None, n_seeds=None):
    envs = " → ".join(ENV_SHORT_NAMES[e] for e in TASKS[task])
    seeds = f", mean ± std of {n_seeds} seeds" if n_seeds else (f", seed {seed}" if seed is not None else "")
    return (f"Continual learning with PPO: {envs}, {rounds} rounds\n"
            f"({len(TASKS[task]) * rounds} phases of {timesteps_per_phase / 1e6:g}M steps, lr {lr:g}{seeds})")


def unique_path(path):
    """path, or path with _2, _3 ... before the extension if it exists already (never overwrite an older file)."""
    root, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(path):
        k += 1
        path = f"{root}_{k}{ext}"
    return path


def run_phases(lr, models_dir, phases, timesteps_per_phase=TIMESTEPS_PER_PHASE, num_env_runners=NUM_ENV_RUNNERS,
               monitor=None, env_settings=None, seed=None):
    """Train all phases with one learning rate; returns (test reward of every phase, verdict of the monitor or
    None if the task went through). The remaining phases are skipped if the monitor stops the task.
    The training curve goes to TensorBoard (models_dir/tb), the x axis is the environment steps cumulated over the
    phases (every phase builds a new algorithm, whose step counter starts at 0): episode_return_mean (sum of the
    agents' scaled rewards) and team_return (minus the total cost, original cost units) of every iteration.
    env_settings: env_config settings of all the phases, e.g. {"max_stockout_days": 0}.
    seed: random seed of the networks and of the demand (RLlib config.seed), the same in every phase."""
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(models_dir, "tb"))
    checkpoint = None
    offset = 0
    test_rewards = {}
    verdict = None
    for i, (phase_name, env_id) in enumerate(phases, start=1):
        env_config = {"env_class": env_id, "obs_rows": OBS_ROWS, **(env_settings or {})}
        config = build_ppo_config(env_config, lr=lr, num_env_runners=num_env_runners, num_gpus=NUM_GPUS, seed=seed)
        algo = config.build_algo()

        # change env: continue from the policies trained in the previous phase
        if checkpoint is not None:
            algo.restore_from_path(os.path.join(checkpoint, RL_MODULE_PATH),
                                   component=RL_MODULE_PATH.replace(os.sep, "/"))
            algo.env_runner_group.sync_weights(from_worker_or_learner_group=algo.learner_group, inference_only=True)

        # train
        tag = f"phase{i:02d}_{ENV_SHORT_NAMES[env_id]}"
        steps = 0
        if monitor:
            monitor.start_phase(i)
        while steps < timesteps_per_phase:
            result = algo.train()
            steps = result["env_runners"]["num_env_steps_sampled_lifetime"]
            ret = result["env_runners"].get("episode_return_mean")
            team = result["env_runners"].get("team_return")
            if ret is not None:
                writer.add_scalar(f"{tag}/episode_return_mean", ret, offset + steps)
            if team is not None:
                writer.add_scalar(f"{tag}/team_return", team, offset + steps)
            print(f"[lr={lr} {i} {phase_name}] steps={steps} episode_return_mean={ret} team_return={team}")
            verdict = monitor.check(result, steps, timesteps_per_phase) if monitor else None
            if verdict:
                verdict["phase_name"] = phase_name
                print(f"[lr={lr} {i} {phase_name}] STOPPED ({verdict['status']}): {verdict['reason']}")
                break
        offset += steps
        writer.flush()
        if verdict and verdict["status"] == "diverged":
            algo.stop()   # no checkpoint / test of a diverged policy
            break

        checkpoint = os.path.join(models_dir, f"PPO_model{i}")
        algo.save_to_path(checkpoint)

        # test the trained model
        test_rewards[phase_name] = evaluate(algo, env_config, n_steps=1000)
        writer.add_scalar("test/episode_return_mean", test_rewards[phase_name], offset)
        print(f"[lr={lr} {i} {phase_name}] test mean episode reward: {test_rewards[phase_name]}")
        algo.stop()
        if verdict:
            break
    writer.close()
    return test_rewards, verdict


def build_parser(prog=None):
    """Command line of masc-continual (also used by the web interface, masc.ui, to build its form)."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--task", required=True, choices=sorted(TASKS), help="order of the environments")
    parser.add_argument("--rounds", type=int, default=None,
                        help="the order of the task is repeated this many times (default: 3, 5 for bat10-sto1 / "
                             "sto1-bat10)")
    parser.add_argument("--out-dir", default=".", help="directory of the models and figures (default: current)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed of the networks and of the demand (default: not fixed); several seeds: one "
                             "run per seed with the same --run-name, then masc-plot-continual --seeds")
    parser.add_argument("--run-name", default=None,
                        help="directory of the run under <models dir>/<task>/ (default: date-time); the runs of the "
                             "seeds of one experiment share it, each one in seed=<seed>/")
    parser.add_argument("--no-figure", action="store_true",
                        help="no figure at the end (e.g. one run per seed, the figure of all the seeds comes after)")
    parser.add_argument("--lr", type=float, nargs="+", default=LRS, help="learning rate(s); several -> grid search")
    parser.add_argument("--timesteps-per-phase", type=int, default=TIMESTEPS_PER_PHASE, help="decisions per phase")
    parser.add_argument("--max-stockout-days", type=int, default=None,
                        help="terminate the episodes after this many stockout days (default: environment value 12)")
    parser.add_argument("--no-stockout-termination", dest="max_stockout_days", action="store_const", const=0,
                        help="never terminate on stockouts: every episode runs the T days (as masc-train)")
    parser.add_argument("--num-env-runners", type=int, default=NUM_ENV_RUNNERS, help="parallel sampling workers")
    parser.add_argument("--no-monitor", action="store_true", help="do not stop a task with a bad reward curve")
    parser.add_argument("--monitor-window", type=int, default=10, help="monitor: iterations per moving window")
    parser.add_argument("--collapse-tol", type=float, default=0.2,
                        help="collapse: the last window is more than this fraction below the best window of the phase")
    parser.add_argument("--collapse-patience", type=int, default=20,
                        help="collapse: iterations the return must stay collapsed before stopping")
    parser.add_argument("--min-gain", type=float, default=0.05,
                        help="no learning: the first phase must improve by this fraction")
    parser.add_argument("--no-learning-steps", type=int, default=400000,
                        help="no learning: checked after this many steps of the first phase (at most the phase length)")
    return parser


def main(argv=None):
    """masc-continual: exit code 1 if the task was stopped for every learning rate."""
    args = build_parser().parse_args(argv)
    if args.rounds is None:
        args.rounds = ROUNDS[args.task]

    phases = make_phases(args.task, args.rounds)
    print(f"task {args.task}, {args.rounds} round(s):", " -> ".join(name for name, _ in phases))

    # Set up directories for saving models and logs.
    stamp = args.run_name or f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
    models_dir = os.path.abspath(os.path.join(args.out_dir, MODELS_DIRS[args.task], args.task, stamp,
                                              "" if args.seed is None else f"seed={args.seed}"))
    os.makedirs(models_dir, exist_ok=True)

    init_ray()

    results, verdicts = {}, {}
    for lr in args.lr:
        lr_dir = os.path.join(models_dir, f"lr={lr}")
        monitor = None if args.no_monitor else ContinualMonitor(
            args.monitor_window, args.collapse_tol, args.collapse_patience, args.min_gain, args.no_learning_steps)
        env_settings = {} if args.max_stockout_days is None else {"max_stockout_days": args.max_stockout_days}
        results[lr], verdict = run_phases(lr, lr_dir, phases, args.timesteps_per_phase, args.num_env_runners, monitor,
                                          env_settings, seed=args.seed)
        verdicts[lr] = verdict
        with open(os.path.join(lr_dir, "continual_status.json"), "w") as f:
            json.dump({"task": args.task, "rounds": args.rounds, "lr": lr, "seed": args.seed, "run_name": stamp,
                       "timesteps_per_phase": args.timesteps_per_phase,
                       "env_settings": env_settings, "phases": [n for n, _ in phases],
                       "status": verdict["status"] if verdict else "completed", "verdict": verdict,
                       "test_rewards": results[lr]}, f, indent=2)
        # printed after every learning rate so that partial results are kept in the job output
        print(f"lr={lr} test rewards:", results[lr])
        if args.no_figure:
            continue
        # reward curves of the task (only at the end, nothing during the training)
        envs = [ENV_SHORT_NAMES[e] for e in TASKS[args.task]]
        stopped = f"_STOPPED-{verdict['status']}" if verdict else ""
        fig = unique_path(os.path.join(args.out_dir, figure_name(args.task, args.rounds, args.timesteps_per_phase, lr,
                                                                 stamp, seed=args.seed, suffix=stopped)))
        title = figure_title(args.task, args.rounds, args.timesteps_per_phase, lr, seed=args.seed)
        if verdict:
            title += f"\nSTOPPED in phase {verdict['phase']} ({verdict['phase_name']}): {verdict['status']}"
        try:
            plot_continual(os.path.join(lr_dir, "tb"), fig, title=title, phases_per_round=len(envs))
            shutil.copy(fig, lr_dir)
        except (FileNotFoundError, KeyError) as e:   # stopped before any finished episode
            print("no figure:", e)
    print("learning rate search results (test reward per phase):")
    for lr, test_rewards in results.items():
        v = verdicts[lr]
        state = f"STOPPED in phase {v['phase']} ({v['status']})" if v else "completed"
        print(f"  lr={lr} [{state}]: " + ", ".join(f"{name} {r:.1f}" for name, r in test_rewards.items()))
    return 0 if any(v is None for v in verdicts.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
