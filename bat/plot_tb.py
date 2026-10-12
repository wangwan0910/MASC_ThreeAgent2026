# -*- coding: utf-8 -*-
"""
Training curve from the TensorBoard logs of a run (main_ppo.py / main_rppo.py), in the style of PPO_demo.png:
episode return of the supply chain against the environment steps, raw curve (light) and smoothed curve (dark,
TensorBoard smoothing).

The return is env_runners/team_return (minus the total cost of the supply chain in the original cost units, logged
by marl_env.team_return_callback), so that runs with different reward modes / reward scales are comparable; runs
logged before this metric existed fall back to env_runners/episode_return_mean.

Every single-environment figure has the same style: colour of the environment (ENV_COLORS), legend in the lower
right corner, both axes in scientific notation and the same axis limits (XLIM, YLIM) for all environments and
algorithms, so that the figures can be compared side by side.
    XLIM: 0 - 2e6 steps, the training length of the multi-seed runs (all trials have flattened by ~1.5e6 steps in the
          3e6-step runs of 2026-10-12)
    YLIM: -2e6 - 0, the episode returns of the smoke tests without stockout termination go from -1.7e6 (first
          iteration, random policies) to -6e4

    python plot_tb.py tu_logs/Bat10/1791700000 --label "PPO in Bat10" --out PPO_Bat10.png
    python plot_tb.py tu_logs/Bat10/1791700000 --lr 0.0003 --ylim -450000 -50000

If the run is a learning-rate grid search, the best trial (highest return at the end) is plotted, or the one of --lr.
Several seeds (--seeds): all the trials (seeds) of one learning rate in one figure, mean of the seeds (line) and
mean +- standard deviation (light band), in the style of PPO_demo.png:

    python plot_tb.py tu_team_logs/Bat10/1791700000 --seeds --label "PPO in Bat10" --out PPO_Bat10.png
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

RETURN_TAGS = ("env_runners/team_return", "env_runners/episode_return_mean")   # first one found is used
STEP_TAG = "num_env_steps_sampled_lifetime"

# one colour per environment, the same in every figure of the project
ENV_COLORS = {"Bat10": "darkgoldenrod", "Bat7": "goldenrod", "Bat3": "gold",
              "Sto0": "dodgerblue", "Sto01": "lightskyblue", "Sto1": "skyblue"}
DEFAULT_COLOR = "gray"
# common axis limits of all single-environment training figures, see the module docstring
XLIM = (0, 2e6)
YLIM = (-2e6, 0)


def env_color(env):
    return ENV_COLORS.get(env, DEFAULT_COLOR)


def env_from_path(path):
    """Environment short name in a log path, e.g. tu_logs/Bat10/<time> -> Bat10 (None if there is none)."""
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    return next((p for p in reversed(parts) if p in ENV_COLORS), None)


def sci_axes(ax):
    """Both axes in scientific notation (e.g. x 1e6 above the axis), also for small values."""
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)


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


def _return_scalars(accumulator):
    for tag in RETURN_TAGS:
        try:
            return _scalars(accumulator, tag)
        except KeyError:
            pass
    raise KeyError(f"no TensorBoard scalar ending with any of {RETURN_TAGS}")


def read_trial(trial_dir):
    """(environment steps, episode return) of one trial, NaN entries (no finished episode yet) removed."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(trial_dir, size_guidance={"scalars": 0})
    acc.Reload()
    returns, steps = _return_scalars(acc), _scalars(acc, STEP_TAG)
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


def trial_seed(trial_dir):
    try:
        with open(os.path.join(trial_dir, "params.json")) as f:
            return json.load(f).get("seed")
    except (OSError, ValueError):
        return None


def seed_curves(logdir, lr=None):
    """(steps, returns of shape (n_seeds, n_steps), seeds) of all the trials of one learning rate (default: the
    learning rate with the best mean final return), on the environment steps common to all the trials (every
    iteration samples the same number of steps, so the trials have the same steps)."""
    trials = find_trials(logdir)
    if not trials:
        raise FileNotFoundError(f"no TensorBoard logs under {logdir}")
    data = {t: read_trial(t) for t in trials}
    data = {t: d for t, d in data.items() if len(d[1])}
    by_lr = {}
    for t in data:
        by_lr.setdefault(trial_lr(t), []).append(t)
    if lr is None:
        def final_return(ts):
            return np.mean([np.mean(data[t][1][-max(1, len(data[t][1]) // 10):]) for t in ts])
        lr = max(by_lr, key=lambda k: final_return(by_lr[k]))
    group = [t for k, ts in by_lr.items() if k is not None and math.isclose(k, lr) for t in ts]
    if not group:
        raise ValueError(f"no trial with lr={lr} under {logdir}")
    group.sort(key=lambda t: (trial_seed(t) is None, trial_seed(t)))
    common = sorted(set.intersection(*(set(data[t][0].tolist()) for t in group)))
    returns = np.array([[dict(zip(data[t][0].tolist(), data[t][1]))[s] for s in common] for t in group])
    return np.array(common), returns, [trial_seed(t) for t in group], lr


def plot_seeds(logdir, label, out, lr=None, xlim=XLIM, ylim=YLIM, env=None, smoothing=0.0):
    """Mean of the seeds (line) and mean +- standard deviation over the seeds (light band) of one learning rate.
    smoothing > 0: TensorBoard smoothing of every seed before the mean (default: raw curves, as PPO_demo.png)."""
    steps, returns, seeds, lr = seed_curves(logdir, lr)
    if smoothing > 0:
        returns = np.array([smooth(r, smoothing) for r in returns])
    mean, std = returns.mean(axis=0), returns.std(axis=0)
    color = env_color(env or env_from_path(logdir))
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.25, linewidth=0)
    ax.plot(steps, mean, color=color, linewidth=1.0, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Episode return")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    sci_axes(ax)
    ax.legend(loc="lower right")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"figure {out}: lr={lr}, {len(seeds)} seeds {seeds}, {len(steps)} points, "
          f"{int(steps[-1]) if len(steps) else 0} steps, final mean {mean[-50:].mean():.4g} +- {std[-50:].mean():.3g}")
    if len(steps) and (steps[-1] > xlim[1] or (mean - std).min() < ylim[0] or (mean + std).max() > ylim[1]):
        print(f"WARNING: the curve goes beyond the common axis limits x {xlim}, y {ylim}")
    return out, (steps, mean, std, seeds, lr)


def plot_curve(steps, returns, label, out, smoothing=0.6, xlim=XLIM, ylim=YLIM, color=DEFAULT_COLOR):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(steps, returns, color=color, alpha=0.25, linewidth=0.8)
    ax.plot(steps, smooth(returns, smoothing), color=color, linewidth=1.0, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Episode return")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    sci_axes(ax)
    ax.legend(loc="lower right")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_run(logdir, label, out, lr=None, smoothing=0.6, xlim=XLIM, ylim=YLIM, env=None):
    """env: environment short name for the colour, default: taken from logdir (e.g. tu_logs/Bat10/<time>)."""
    trial, (steps, returns) = select_trial(logdir, lr)
    env = env or env_from_path(logdir)
    plot_curve(steps, returns, label, out, smoothing, xlim, ylim, color=env_color(env))
    print(f"figure {out}: trial {os.path.basename(trial)} (lr={trial_lr(trial)}), {len(steps)} points, "
          f"{int(steps[-1]) if len(steps) else 0} steps")
    if len(steps) and (steps[-1] > xlim[1] or returns.min() < ylim[0] or returns.max() > ylim[1]):
        print(f"WARNING: the curve goes beyond the common axis limits x {xlim}, y {ylim} "
              f"(steps up to {steps[-1]:.3g}, returns {returns.min():.3g} to {returns.max():.3g})")
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="log directory of a run, e.g. tu_logs/Bat10/<time>")
    parser.add_argument("--label", default="PPO", help='legend, e.g. "PPO in Bat10"')
    parser.add_argument("--out", default="PPO.png")
    parser.add_argument("--lr", type=float, default=None, help="trial to plot (default: the best one)")
    parser.add_argument("--env", default=None, choices=sorted(ENV_COLORS),
                        help="environment for the colour (default: taken from logdir)")
    parser.add_argument("--seeds", action="store_true",
                        help="all the trials (seeds) of one lr in one figure: mean line and +- std band")
    parser.add_argument("--smoothing", type=float, default=None,
                        help="TensorBoard smoothing (default 0.6, with --seeds 0: raw curves)")
    parser.add_argument("--xlim", type=float, nargs=2, default=XLIM, help="default: common limits of all figures")
    parser.add_argument("--ylim", type=float, nargs=2, default=YLIM, help="default: common limits of all figures")
    a = parser.parse_args()
    if a.seeds:
        plot_seeds(a.logdir, a.label, a.out, a.lr, a.xlim, a.ylim, env=a.env, smoothing=a.smoothing or 0.0)
    else:
        plot_run(a.logdir, a.label, a.out, a.lr, 0.6 if a.smoothing is None else a.smoothing, a.xlim, a.ylim,
                 env=a.env)
