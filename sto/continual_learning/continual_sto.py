# -*- coding: utf-8 -*-
"""
Continual learning with multi-agent PPO (Ray RLlib) between the stochastic demand environments of this
directory (AssembySto: std 0 / 0.1 / 1).
Every phase starts from the policies of the previous phase; the sequence of a task is repeated --rounds times.

Tasks (--task):
    0-01-1 : std 0 -> 0.1 -> 1
    1-01-0 : std 1 -> 0.1 -> 0

    python continual_learning/continual_sto.py                                   # task 1-01-0, lr 0.003
    python continual_learning/continual_sto.py --task 0-01-1 --rounds 1
    python continual_learning/continual_sto.py --lr 0.03 0.003 0.0003           # grid search, one lr after another

The training curve (mean episode return per iteration, cumulated environment steps) is written to TensorBoard,
<models_dir>/tb/, one scalar per phase, and plotted at the end of the task (one figure per learning rate, in the
style of dcontinustoPPO2_2.png): continual_learning/PPOcon_<task>_r<rounds>_lr<lr>_<date-time>.png (a copy stays in
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

# marl_env.py is in the parent directory of this script, plot_continual.py next to it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marl_env import ENV_SHORT_NAMES, build_ppo_config, evaluate, init_ray
from plot_continual import plot_continual


OBS_ROWS = 9
LRS = [0.003]
TIMESTEPS_PER_PHASE = 200000
NUM_ENV_RUNNERS = 1
NUM_GPUS = 0
ROUNDS = 3

# continual learning tasks: order of the environments, repeated ROUNDS times
# StoSupplyEnv-v1 = std 0 (demand always 2), StoSupplyEnv-v2 = std 0.1, StoSupplyEnv-v3 = std 1 (see marl_env.ENV_IDS)
TASKS = {
    "0-01-1": ["StoSupplyEnv-v1", "StoSupplyEnv-v2", "StoSupplyEnv-v3"],   # std 0 -> 0.1 -> 1
    "1-01-0": ["StoSupplyEnv-v3", "StoSupplyEnv-v2", "StoSupplyEnv-v1"],   # std 1 -> 0.1 -> 0
}
DEFAULT_TASK = "1-01-0"

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = "PPOcon-sto_models"
RL_MODULE_PATH = os.path.join(COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER, COMPONENT_RL_MODULE)


def make_phases(task, rounds=ROUNDS):
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


def unique_path(path):
    """path, or path with _2, _3 ... before the extension if it exists already (never overwrite an older file)."""
    root, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(path):
        k += 1
        path = f"{root}_{k}{ext}"
    return path


def run_phases(lr, models_dir, phases, timesteps_per_phase=TIMESTEPS_PER_PHASE, num_env_runners=NUM_ENV_RUNNERS,
               monitor=None):
    """Train all phases with one learning rate; returns (test reward of every phase, verdict of the monitor or
    None if the task went through). The remaining phases are skipped if the monitor stops the task.
    The training curve goes to TensorBoard (models_dir/tb), the x axis is the environment steps cumulated over the
    phases (every phase builds a new algorithm, whose step counter starts at 0)."""
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(models_dir, "tb"))
    checkpoint = None
    offset = 0
    test_rewards = {}
    verdict = None
    for i, (phase_name, env_id) in enumerate(phases, start=1):
        env_config = {"env_class": env_id, "obs_rows": OBS_ROWS}
        config = build_ppo_config(env_config, lr=lr, num_env_runners=num_env_runners, num_gpus=NUM_GPUS)
        algo = config.build_algo()

        # change env: continue from the policies trained in the previous phase
        if checkpoint is not None:
            algo.restore_from_path(os.path.join(checkpoint, RL_MODULE_PATH),
                                   component=RL_MODULE_PATH.replace(os.sep, "/"))
            algo.env_runner_group.sync_weights(from_worker_or_learner_group=algo.learner_group, inference_only=True)

        # train
        tag = f"phase{i:02d}_{ENV_SHORT_NAMES[env_id]}/episode_return_mean"
        steps = 0
        if monitor:
            monitor.start_phase(i)
        while steps < timesteps_per_phase:
            result = algo.train()
            steps = result["env_runners"]["num_env_steps_sampled_lifetime"]
            ret = result["env_runners"].get("episode_return_mean")
            if ret is not None:
                writer.add_scalar(tag, ret, offset + steps)
            print(f"[lr={lr} {i} {phase_name}] steps={steps} episode_return_mean={ret}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=DEFAULT_TASK, choices=sorted(TASKS), help="order of the environments")
    parser.add_argument("--rounds", type=int, default=ROUNDS, help="the order of the task is repeated this many times")
    parser.add_argument("--lr", type=float, nargs="+", default=LRS, help="learning rate(s); several -> grid search")
    parser.add_argument("--timesteps-per-phase", type=int, default=TIMESTEPS_PER_PHASE, help="decisions per phase")
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
    args = parser.parse_args()

    phases = make_phases(args.task, args.rounds)
    print(f"task {args.task}, {args.rounds} round(s):", " -> ".join(name for name, _ in phases))

    # Set up directories for saving models and logs.
    stamp = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
    models_dir = os.path.abspath(os.path.join(MODELS_DIR, args.task, stamp))
    os.makedirs(models_dir, exist_ok=True)

    init_ray()

    results, verdicts = {}, {}
    for lr in args.lr:
        lr_dir = os.path.join(models_dir, f"lr={lr}")
        monitor = None if args.no_monitor else ContinualMonitor(
            args.monitor_window, args.collapse_tol, args.collapse_patience, args.min_gain, args.no_learning_steps)
        results[lr], verdict = run_phases(lr, lr_dir, phases, args.timesteps_per_phase, args.num_env_runners, monitor)
        verdicts[lr] = verdict
        with open(os.path.join(lr_dir, "continual_status.json"), "w") as f:
            json.dump({"task": args.task, "rounds": args.rounds, "lr": lr, "phases": [n for n, _ in phases],
                       "status": verdict["status"] if verdict else "completed", "verdict": verdict,
                       "test_rewards": results[lr]}, f, indent=2)
        # printed after every learning rate so that partial results are kept in the job output
        print(f"lr={lr} test rewards:", results[lr])
        # reward curves of the task, e.g. PPOcon_1-01-0_r3_lr0.003_20261011-170501.png
        stopped = f"_STOPPED-{verdict['status']}" if verdict else ""
        fig = unique_path(os.path.join(FIG_DIR, f"PPOcon_{args.task}_r{args.rounds}_lr{lr:g}_{stamp}{stopped}.png"))
        title = f"Continuous agent reward curves for PPO ({args.task}, lr {lr:g})"
        if verdict:
            title += f"\nSTOPPED in phase {verdict['phase']} ({verdict['phase_name']}): {verdict['status']}"
        try:
            plot_continual(os.path.join(lr_dir, "tb"), fig, title=title)
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
