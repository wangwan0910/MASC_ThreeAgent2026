# -*- coding: utf-8 -*-
"""
Reward curves of a continual learning run from its TensorBoard logs, in the style of
dcontinustoPPO2_2.png: mean episode return against the environment steps (cumulated over the phases),
one segment per phase, coloured by environment.

The continual learning script writes one scalar per phase, "phase<i>_<env>/episode_return_mean"
(e.g. phase01_Sto1/episode_return_mean), with the cumulated environment steps as step.

    python plot_continual.py <models_dir>/tb --out PPOcon.png
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = re.compile(r"^phase(\d+)_(\w+)/episode_return_mean$")
# one colour per environment, the same in every figure
COLORS = {"Sto1": "#5DADE2", "Sto01": "#48C9B0", "Sto0": "#F5B041",
          "Bat10": "#AF7AC5", "Bat7": "#EC7063", "Bat3": "#58D68D"}


def read_phases(tb_dir):
    """[(phase number, env short name, steps, returns)] sorted by phase."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    if not glob.glob(os.path.join(tb_dir, "events.out.tfevents.*")):
        raise FileNotFoundError(f"no TensorBoard logs in {tb_dir}")
    acc = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    acc.Reload()
    phases = []
    for tag in acc.Tags()["scalars"]:
        m = TAG.match(tag)
        if m:
            events = acc.Scalars(tag)
            phases.append((int(m.group(1)), m.group(2), [e.step for e in events], [e.value for e in events]))
    if not phases:
        raise KeyError(f"no 'phase<i>_<env>/episode_return_mean' scalar in {tb_dir}")
    return sorted(phases)


def plot_continual(tb_dir, out, title="Continuous agent reward curves for PPO"):
    fig, ax = plt.subplots(figsize=(12, 4))
    labelled = set()
    for _, env, steps, returns in read_phases(tb_dir):
        ax.plot(steps, returns, color=COLORS.get(env), linewidth=0.6,
                label=None if env in labelled else env)
        labelled.add(env)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward history")
    ax.legend(loc="upper left")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("figure", out)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tb_dir", help="TensorBoard directory of a continual learning run, e.g. <models_dir>/tb")
    parser.add_argument("--out", default="PPOcon.png")
    parser.add_argument("--title", default="Continuous agent reward curves for PPO")
    a = parser.parse_args()
    plot_continual(a.tb_dir, a.out, a.title)
