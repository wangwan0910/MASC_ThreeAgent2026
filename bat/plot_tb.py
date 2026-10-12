# -*- coding: utf-8 -*-
"""
Training curve from the TensorBoard logs of a run (main_ppo.py / main_rppo.py), in the style of PPO_demo.png:
mean episode return of the supply chain (sum of the three agents' rewards) against the environment steps,
raw curve (light) and smoothed curve (dark, TensorBoard smoothing).

    python plot_tb.py tu_logs/Bat10/1791700000 --label "PPO in Bat10" --out PPO_Bat10.png
    python plot_tb.py tu_logs/Bat10/1791700000 --lr 0.003 --ylim -450000 -50000

If the run is a learning-rate grid search, the best trial (highest return at the end) is plotted, or the one of --lr.
"""

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RETURN_TAG = "env_runners/episode_return_mean"
STEP_TAG = "num_env_steps_sampled_lifetime"


def find_trials(logdir):
    """Trial directories (containing TensorBoard event files) under logdir."""
    files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)
    return sorted({os.path.dirname(f) for f in files})


def trial_lr(trial_dir):
    try:
        with open(os.path.join(trial_dir, "params.json")) as f:
            return json.load(f).get("lr")
    except (OSError, ValueError):
        return None


def _scalars(accumulator, suffix):
    tags = [t for t in accumulator.Tags()["scalars"] if t.endswith(suffix)]
    if not tags:
        raise KeyError(f"no TensorBoard scalar ending with '{suffix}'")
    return {e.step: e.value for e in accumulator.Scalars(min(tags, key=len))}


def read_trial(trial_dir):
    """(environment steps, mean episode return) of one trial, NaN entries (no finished episode yet) removed."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(trial_dir, size_guidance={"scalars": 0})
    acc.Reload()
    returns, steps = _scalars(acc, RETURN_TAG), _scalars(acc, STEP_TAG)
    it = [i for i in sorted(returns) if i in steps and not math.isnan(returns[i])]
    return np.array([steps[i] for i in it]), np.array([returns[i] for i in it])


def smooth(values, weight=0.6):
    """TensorBoard smoothing (debiased exponential moving average)."""
    out, last, n = [], 0.0, 0
    for v in values:
        last = last * weight + (1 - weight) * v
        n += 1
        out.append(last / (1 - weight ** n))
    return np.array(out)


def select_trial(logdir, lr=None):
    trials = find_trials(logdir)
    if not trials:
        raise FileNotFoundError(f"no TensorBoard logs under {logdir}")
    if lr is not None:
        trials = [t for t in trials if trial_lr(t) is not None and math.isclose(trial_lr(t), lr)]
        if not trials:
            raise ValueError(f"no trial with lr={lr} under {logdir}")
    data = {t: read_trial(t) for t in trials}
    data = {t: d for t, d in data.items() if len(d[1])}

    def final_return(t):
        r = data[t][1]
        return np.mean(r[-max(1, len(r) // 10):])
    best = max(data, key=final_return)
    return best, data[best]


def plot_curve(steps, returns, label, out, smoothing=0.6, ylim=None, color="darkgoldenrod"):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(steps, returns, color=color, alpha=0.25, linewidth=0.8)
    ax.plot(steps, smooth(returns, smoothing), color=color, linewidth=1.0, label=label)
    ax.set_xlabel("Step")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper right")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_run(logdir, label, out, lr=None, smoothing=0.6, ylim=None):
    trial, (steps, returns) = select_trial(logdir, lr)
    plot_curve(steps, returns, label, out, smoothing, ylim)
    print(f"figure {out}: trial {os.path.basename(trial)} (lr={trial_lr(trial)}), {len(steps)} points, "
          f"{int(steps[-1]) if len(steps) else 0} steps")
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="log directory of a run, e.g. tu_logs/Bat10/<time>")
    parser.add_argument("--label", default="PPO", help='legend, e.g. "PPO in Bat10"')
    parser.add_argument("--out", default="PPO.png")
    parser.add_argument("--lr", type=float, default=None, help="trial to plot (default: the best one)")
    parser.add_argument("--smoothing", type=float, default=0.6)
    parser.add_argument("--ylim", type=float, nargs=2, default=None)
    a = parser.parse_args()
    plot_run(a.logdir, a.label, a.out, a.lr, a.smoothing, a.ylim)
