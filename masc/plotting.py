# -*- coding: utf-8 -*-
"""
Figures of the masc package (masc-plot, masc-plot-continual).

Training curve from the TensorBoard logs of a run (masc-train / masc-train-transformer), in the style of
assets/PPO_demo.png:
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

    masc-plot tu_logs/Bat10/1791700000 --label "PPO in Bat10" --out PPO_Bat10.png
    masc-plot tu_logs/Bat10/1791700000 --lr 0.0003 --ylim -450000 -50000

If the run is a learning-rate grid search, the best trial (highest return at the end) is plotted, or the one of --lr.
Several seeds (--seeds): all the trials (seeds) of one learning rate in one figure, mean of the seeds (line) and
mean +- standard deviation (light band), in the style of PPO_demo.png:

    masc-plot tu_team_logs/Bat10/1791700000 --seeds --label "PPO in Bat10" --out PPO_Bat10.png

Continual learning (masc-continual): reward curves of all the phases of a task, one colour per environment, in the
style of assets/dcontinustoPPO2_2.png (plot_continual):

    masc-plot-continual <models_dir>/lr=<lr>/tb --out PPOcon.png
"""

import argparse
import glob
import json
import math
import os
import re

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
# resolution of the continual learning figures (wide, with a detail inset: the default 100 dpi is too coarse)
CONTINUAL_DPI = 300


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
    files = glob.glob(os.path.join(glob.escape(logdir), "**", "events.out.tfevents.*"), recursive=True)
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
    learning rate with the best mean final return), interpolated on a common step grid (align_curves)."""
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
    steps, returns = align_curves([data[t] for t in group])
    return steps, returns, [trial_seed(t) for t in group], lr


def align_curves(curves):
    """Curves [(steps, values)] of several trials on one step grid: the steps of the first trial within the range
    covered by all of them, the values linearly interpolated. (An iteration samples a little more than
    train_batch_size steps now and then, e.g. 3072 instead of 2048, so the step values of the trials drift apart and
    keeping only the steps common to all trials would cut the curves.) Returns (steps, values (n_trials, n_steps))."""
    lo = max(float(s[0]) for s, _ in curves)
    hi = min(float(s[-1]) for s, _ in curves)
    grid = np.asarray(curves[0][0], dtype=float)
    grid = grid[(grid >= lo) & (grid <= hi)]
    return grid, np.array([np.interp(grid, np.asarray(s, dtype=float), np.asarray(v, dtype=float))
                           for s, v in curves])


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
    if len(steps) and (steps[-1] > xlim[1] * 1.01 or (mean - std).min() < ylim[0] or (mean + std).max() > ylim[1]):
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
    if len(steps) and (steps[-1] > xlim[1] * 1.01 or returns.min() < ylim[0] or returns.max() > ylim[1]):
        print(f"WARNING: the curve goes beyond the common axis limits x {xlim}, y {ylim} "
              f"(steps up to {steps[-1]:.3g}, returns {returns.min():.3g} to {returns.max():.3g})")
    return out


TAG = re.compile(r"^phase(\d+)_(\w+)/(episode_return_mean|team_return)$")
METRIC_LABELS = {"team_return": "Episode return (minus total cost)", "episode_return_mean": "Reward history (scaled)"}


def read_phases(tb_dir):
    """Phases of a continual learning run: [(phase number, env short name, steps, returns)] sorted by phase."""
    return read_phases_metric(tb_dir)[0]


def read_phases_metric(tb_dir):
    """(phases, metric): the continual learning writes per phase "phase<i>_<env>/team_return" (minus the total cost,
    original cost units) and "phase<i>_<env>/episode_return_mean" (scaled rewards; the only one of older runs), with the
    cumulated environment steps as step; team_return is used when it exists."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    if not glob.glob(os.path.join(glob.escape(tb_dir), "events.out.tfevents.*")):
        raise FileNotFoundError(f"no TensorBoard logs in {tb_dir}")
    acc = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    acc.Reload()
    found = {}
    for tag in acc.Tags()["scalars"]:
        m = TAG.match(tag)
        if m:
            events = acc.Scalars(tag)
            found.setdefault(m.group(3), []).append(
                (int(m.group(1)), m.group(2), [e.step for e in events], [e.value for e in events]))
    if not found:
        raise KeyError(f"no 'phase<i>_<env>/team_return' or '.../episode_return_mean' scalar in {tb_dir}")
    metric = "team_return" if "team_return" in found else "episode_return_mean"
    return sorted(found[metric]), metric


def _round_mark(ax, number, start, phases_per_round):
    """Dotted line and label "round k" at the start of every round of a continual learning figure."""
    if phases_per_round and (number - 1) % phases_per_round == 0:
        round_number = (number - 1) // phases_per_round + 1
        if round_number > 1:
            ax.axvline(start, color="gray", linewidth=0.6, linestyle=":")
        ax.annotate(f"round {round_number}", (start, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -12), textcoords="offset points", fontsize=8, color="gray")


def _finish_continual(fig, ax, out, title, metric):
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(METRIC_LABELS[metric])
    sci_axes(ax)
    ax.legend(loc="lower right")
    fig.savefig(out, bbox_inches="tight", dpi=CONTINUAL_DPI)
    plt.close(fig)
    print("figure", out)
    return out


def plot_continual(tb_dir, out, title="Continuous agent reward curves for PPO", phases_per_round=None):
    """Reward curves of a continual learning run: one segment per phase, coloured by environment (ENV_COLORS),
    both axes in scientific notation; phases_per_round: dotted line and label at the start of every round."""
    fig, ax = plt.subplots(figsize=(12, 4))
    phases, metric = read_phases_metric(tb_dir)
    labelled = set()
    for number, env, steps, returns in phases:
        ax.plot(steps, returns, color=ENV_COLORS.get(env), linewidth=0.6,
                label=None if env in labelled else env)
        labelled.add(env)
        if steps:
            _round_mark(ax, number, steps[0], phases_per_round)
    return _finish_continual(fig, ax, out, title, metric)


def seed_tb_dirs(run_dir, lr=None):
    """TensorBoard directories of the seeds of a continual learning experiment: <run_dir>/seed=<s>/lr=<lr>/tb
    (masc-continual --seed <s> --run-name <name>), {seed: tb dir}, of the learning rate lr (default: the only one)."""
    found = {}
    for tb in glob.glob(os.path.join(glob.escape(run_dir), "seed=*", "lr=*", "tb")):
        seed = int(os.path.basename(os.path.dirname(os.path.dirname(tb)))[len("seed="):])
        tb_lr = float(os.path.basename(os.path.dirname(tb))[len("lr="):])
        if lr is None or math.isclose(tb_lr, lr):
            found.setdefault(tb_lr, {})[seed] = tb
    if not found:
        raise FileNotFoundError(f"no seed=<s>/lr=<lr>/tb directory under {run_dir}")
    if len(found) > 1:
        raise ValueError(f"several learning rates under {run_dir}: {sorted(found)}, choose one with lr")
    return dict(sorted(next(iter(found.values())).items()))


def continual_seed_phases(run_dir, lr=None):
    """Phases of the seeds of a continual learning experiment: ([(phase number, env, steps, mean, std, n seeds)],
    metric, seeds). The curves of a phase are interpolated on a common step grid (align_curves): the cumulated steps
    of the seeds differ a little (some iterations sample more steps)."""
    per_seed = {seed: read_phases_metric(tb) for seed, tb in seed_tb_dirs(run_dir, lr).items()}
    metrics = {m for _, m in per_seed.values()}
    if len(metrics) > 1:
        raise ValueError(f"the seeds under {run_dir} logged different metrics: {sorted(metrics)}")
    metric = metrics.pop()
    curves = {}
    for phases, _ in per_seed.values():
        for number, env, steps, returns in phases:
            if steps:
                curves.setdefault((number, env), []).append((steps, returns))
    result = []
    for (number, env), cs in sorted(curves.items()):
        grid, values = align_curves(cs)
        result.append((number, env, grid, values.mean(axis=0), values.std(axis=0), len(cs)))
    return result, metric, sorted(per_seed)


def _draw_seed_phases(ax, phases, phases_per_round, legend=True, round_labels=True):
    labelled = set()
    for number, env, steps, mean, std, n in phases:
        color = ENV_COLORS.get(env)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.25, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=0.8, label=None if env in labelled or not legend else env)
        labelled.add(env)
        if len(steps) and round_labels:
            _round_mark(ax, number, steps[0], phases_per_round)
        elif len(steps) and phases_per_round and number > 1 and (number - 1) % phases_per_round == 0:
            ax.axvline(steps[0], color="gray", linewidth=0.6, linestyle=":")


def _zoom_ylim(phases, skip=0.5, margin=0.1):
    """y range of the detail inset: mean +- std of all the phases, without the first half (skip) of the first phase,
    where the policies trained from scratch still have returns far below the rest of the task."""
    lows, highs = [], []
    for i, (number, env, steps, mean, std, n) in enumerate(phases):
        start = int(len(mean) * skip) if i == 0 else 0
        if len(mean) > start:
            lows.append(np.min(mean[start:] - std[start:]))
            highs.append(np.max(mean[start:] + std[start:]))
    lo, hi = min(lows), max(highs)
    return lo - margin * (hi - lo), hi + margin * (hi - lo)


def plot_continual_seeds(run_dir, out, title, phases_per_round=None, lr=None, inset=True):
    """Continual learning, several seeds: mean of the seeds (line) and mean +- std (light band) of every phase,
    coloured by environment, in the style of assets/dcontinustoPPO2_2.png.
    inset: detail of the curves (zoomed y axis, _zoom_ylim) in the empty lower part of the figure, where the +- std
    bands can be seen; drawn only if the zoomed range is less than a quarter of the full y range."""
    phases, metric, seeds = continual_seed_phases(run_dir, lr)
    fig, ax = plt.subplots(figsize=(12, 4))
    _draw_seed_phases(ax, phases, phases_per_round)
    incomplete = [(number, n) for number, _, _, _, _, n in phases if n < len(seeds)]
    if incomplete:
        print(f"WARNING: phases with fewer than {len(seeds)} seeds (phase, seeds): {incomplete}")
    if inset and phases:
        lo, hi = _zoom_ylim(phases)
        full_lo, full_hi = ax.get_ylim()
        if hi - lo < 0.25 * (full_hi - full_lo):
            # below the curves: the main axes are empty there except the start of the first phase (left edge) and
            # the legend (lower right)
            ins = ax.inset_axes([0.1, 0.06, 0.76, 0.62])
            _draw_seed_phases(ins, phases, phases_per_round, legend=False, round_labels=False)
            ins.set_xlim(ax.get_xlim())
            ins.set_ylim(lo, hi)
            ins.tick_params(labelsize=7)
            ins.set_title("detail (mean ± std)", fontsize=8)
            sci_axes(ins)
            ins.yaxis.get_offset_text().set_fontsize(7)
            ins.xaxis.get_offset_text().set_visible(False)
            ax.indicate_inset_zoom(ins, edgecolor="gray")
    return _finish_continual(fig, ax, out, title, metric)


def main_continual(argv=None):
    """masc-plot-continual: reward curves of a continual learning run (one tb directory), or with --seeds of all the
    seeds of an experiment (directory with seed=<s>/lr=<lr>/tb), named and titled from its continual_status.json."""
    parser = argparse.ArgumentParser(prog="masc-plot-continual")
    parser.add_argument("path", help="TensorBoard directory of a run (<models_dir>/lr=<lr>/tb), or with --seeds the "
                                     "directory of the experiment (<models dir>/<task>/<run name>)")
    parser.add_argument("--seeds", action="store_true", help="mean +- std of the seeds of the experiment")
    parser.add_argument("--lr", type=float, default=None, help="--seeds: learning rate (default: the only one)")
    parser.add_argument("--out", default=None, help="figure file (default: PPOcon.png, --seeds: name from the "
                                                    "settings of the experiment, in the current directory)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-inset", action="store_true",
                        help="--seeds: no detail inset (zoomed y axis with the +- std bands) in the empty part")
    a = parser.parse_args(argv)
    if not a.seeds:
        plot_continual(a.path, a.out or "PPOcon.png", a.title or "Continuous agent reward curves for PPO")
        return 0
    from .continual import TASKS, figure_name, figure_title
    tbs = seed_tb_dirs(a.path, a.lr)
    # settings of the experiment from the first seed that finished (a crashed seed has no continual_status.json)
    statuses = [os.path.join(os.path.dirname(tb), "continual_status.json") for tb in tbs.values()]
    finished = [s for s in statuses if os.path.exists(s)]
    if not finished:
        raise SystemExit(f"no continual_status.json in the seeds of {a.path}: no seed has finished")
    with open(finished[0]) as f:
        st = json.load(f)
    name = (st["task"], st["rounds"], st["timesteps_per_phase"], st["lr"])
    algo = st.get("algo", "PPO")
    out = a.out or figure_name(*name, st.get("run_name") or os.path.basename(a.path), n_seeds=len(tbs), algo=algo)
    plot_continual_seeds(a.path, out, a.title or figure_title(*name, n_seeds=len(tbs), algo=algo),
                         phases_per_round=len(TASKS[st["task"]]), lr=a.lr, inset=not a.no_inset)
    return 0


def main(argv=None):
    """masc-plot: training curve of a run (one trial, or mean +- std of the seeds with --seeds)."""
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
    a = parser.parse_args(argv)
    if a.seeds:
        plot_seeds(a.logdir, a.label, a.out, a.lr, a.xlim, a.ylim, env=a.env, smoothing=a.smoothing or 0.0)
    else:
        plot_run(a.logdir, a.label, a.out, a.lr, 0.6 if a.smoothing is None else a.smoothing, a.xlim, a.ylim,
                 env=a.env)
    return 0


# policy figures of a trajectory CSV (masc-trajectory), in the style of assets/OPst01.png (orders) and
# assets/RinvBat10.png (inventory)
POLICY_FIGSIZE = (10, 12)
POLICY_FONTSIZE = 20
# colour of the line that tells the environments apart (reorder point in the order figure, retailer stockout in the
# inventory figure); other environments keep REORDER_POINT_COLOR / STOCKOUT_COLOR
POLICY_ENV_COLORS = {"Sto0": "m", "Sto01": "teal", "Sto1": "darkgreen",
                     "Bat10": "palevioletred", "Bat7": "steelblue", "Bat3": "purple"}
REORDER_POINT_COLOR = "cyan"
STOCKOUT_COLOR = "tomato"


def read_trajectory(csv_path):
    """{seed: {column: np.array over the days}} of a trajectory CSV (masc-trajectory)."""
    import csv
    rows = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.setdefault(int(row["seed"]), []).append(row)
    return {seed: {k: np.array([float(r[k]) for r in rs]) for k in rs[0]} for seed, rs in sorted(rows.items())}


def best_seed(trajectories):
    """Seed whose episode has the highest total Cost (the least total cost of the supply chain)."""
    return max(trajectories, key=lambda seed: trajectories[seed]["Cost"].sum())


def _policy_legend(ax, gap_px=12):
    """Legend inside the axes, lower right corner. The values of the policy figures are >= 0, so the y axis is
    extended below 0 until the legend fits under y = 0 (gap_px pixels of clearance): it covers no line."""
    leg = ax.legend(loc="lower right", ncol=2, fontsize=POLICY_FONTSIZE - 4, framealpha=0.9)
    fig = ax.figure
    fig.canvas.draw()
    ab, lb = ax.get_window_extent(), leg.get_window_extent()
    # fraction of the axes height the legend (and the clearance) takes from the bottom
    r = (lb.y1 - ab.y0 + gap_px) / ab.height
    top = ax.get_ylim()[1]
    ax.set_ylim(-r * top / (1 - r), top)
    return leg


def _policy_axes(title):
    plt.rcParams.update({"font.size": POLICY_FONTSIZE})
    fig, ax = plt.subplots(figsize=POLICY_FIGSIZE)
    ax.set_title(title)
    ax.set_xlabel("Step")
    return fig, ax


def plot_policy_orders(traj, title, out, days=30):
    """Orders of the three agents and reorder point of the retailer over the first days of the episode
    (assets/OPst01.png). The thin lines are drawn above the thick ones: the orders of the agents are often equal.
    The reorder point is only set on reorder days and kept in between: step line, markers on the reorder days."""
    fig, ax = _policy_axes(title)
    t = traj["day"][:days]
    ax.plot(t, traj["retailer_orderToSupplier"][:days], color="blue", linewidth=1, label="retailer order", zorder=4)
    ax.plot(t, traj["warehouse_orderToSupplier"][:days], color="darkgoldenrod", linestyle="--", linewidth=2,
            label="warehouse order", zorder=3)
    ax.plot(t, traj["factory_orderToSupplier"][:days], color="silver", linewidth=3, label="factory order", zorder=2)
    reorder_days = traj["retailer_actionTriggerHistory"][:days] > 0
    ax.step(t, traj["retailer_reorderpoint"][:days], where="post",
            color=POLICY_ENV_COLORS.get(title, REORDER_POINT_COLOR), linewidth=2, marker="o", markersize=6,
            markevery=np.flatnonzero(reorder_days).tolist(), label="reorder point")
    ax.set_ylim(-5, 32)
    _policy_legend(ax)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("figure", out)
    return out


def stockout_before_execute(traj, agent="retailer"):
    """Cumulative stockouts of the agent before the actions are executed: stockouts since the last reorder day,
    the current day included (the CSV state is the state after execute, where stockout_cum is reset to 0 on the
    reorder days). On a reorder day this is the stockout charged in the cost."""
    daily, trigger = traj[f"{agent}_stockout_daily"], traj["retailer_actionTriggerHistory"]
    cum, total = np.zeros_like(daily), 0.0
    for i, (d, reorder) in enumerate(zip(daily, trigger)):
        total += d
        cum[i] = total
        if reorder:
            total = 0.0
    return cum


def plot_policy_inventory(traj, title, out, days=100):
    """Inventory of the three agents and cumulative stockout of the retailer before execute (stockouts since the
    last reorder day, see stockout_before_execute), step lines from day 0 (initial inventory taken as the
    inventory of day 1 + its demand is not known, so the lines start at 0 as in assets/RinvBat10.png)."""
    fig, ax = _policy_axes(title)
    t = np.concatenate([[0], traj["day"][:days]])

    def steps(column):
        return np.concatenate([[0], traj[column][:days]])

    ax.step(t, steps("retailer_inventory"), where="post", color="blue", linewidth=1.5, label="retailer inventory")
    ax.step(t, steps("warehouse_inventory"), where="post", color="orange", linestyle="--", linewidth=2.5,
            label="warehouse inventory")
    ax.step(t, steps("factory_inventory"), where="post", color="gray", linewidth=1.5, label="factory inventory")
    stockout = np.concatenate([[0], stockout_before_execute(traj)[:days]])
    ax.step(t, stockout, where="post", color=POLICY_ENV_COLORS.get(title, STOCKOUT_COLOR),
            linewidth=2.5, marker="^", markersize=6, label="retailer cumulative stockout")
    ax.set_ylim(-8, 32)
    _policy_legend(ax)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("figure", out)
    return out


def main_policy(argv=None):
    """masc-plot-policy: order and inventory figures of the trained policies, from a trajectory CSV."""
    parser = argparse.ArgumentParser(prog="masc-plot-policy")
    parser.add_argument("csv", help="trajectory CSV (masc-trajectory), e.g. sto/csv_results/PPO_Sto0_..._trajectory.csv")
    parser.add_argument("--seed", type=int, default=None, help="seed of the episode (default: the least total cost)")
    parser.add_argument("--env", default=None, help="title, e.g. Sto0 (default: taken from the file name)")
    parser.add_argument("--prefix", default="PPO", help="start of the figure names (default: PPO)")
    parser.add_argument("--order-days", type=int, default=30, help="days of the order figure (default: 30)")
    parser.add_argument("--inventory-days", type=int, default=100, help="days of the inventory figure (default: 100)")
    parser.add_argument("--out-dir", default=".", help="directory of the figures (default: current)")
    a = parser.parse_args(argv)
    trajectories = read_trajectory(a.csv)
    seed = best_seed(trajectories) if a.seed is None else a.seed
    env = a.env or env_from_path(os.path.basename(a.csv).replace("_", os.sep)) or "policy"
    costs = ", ".join(f"{s}: {-tr['Cost'].sum():g}" for s, tr in trajectories.items())
    print(f"seed {seed}, total cost {-trajectories[seed]['Cost'].sum():g} (all seeds: {costs})")
    os.makedirs(a.out_dir, exist_ok=True)
    plot_policy_orders(trajectories[seed], env, os.path.join(a.out_dir, f"{a.prefix}_OP{env}_seed{seed}.png"),
                       a.order_days)
    plot_policy_inventory(trajectories[seed], env, os.path.join(a.out_dir, f"{a.prefix}_Rinv{env}_seed{seed}.png"),
                          a.inventory_days)
    return 0


if __name__ == '__main__':
    main()
