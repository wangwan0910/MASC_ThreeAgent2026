# -*- coding: utf-8 -*-
"""
Multi-agent PPO (Ray RLlib) on the stationary / stochastic supply chain environment.
Every stage (factory, warehouse, retailer) is an agent with its own PPO policy, see marl_env.py.

The reward curve of every trial is checked during the training (TrainingMonitor):
    converged  : the return has stayed stable (two consecutive windows of --conv-window iterations within
                 --conv-tol + noise) and near its best for --conv-patience iterations, and has improved by at
                 least --min-gain since the start -> the trial stops, healthy
    diverged   : non-finite return or loss -> stopped at once
    collapsed  : the return stays far below its best (--collapse-tol) for --collapse-patience iterations -> stopped
    stalled    : no improvement of --conv-tol for --stall-patience iterations and not converged -> stopped
    no_learning: same as stalled, but the return never improved by --min-gain since the start
    max_timesteps / time_budget: --total-timesteps or --time-budget-hours reached before convergence
    completed  : --no-early-stop: the trial runs the --total-timesteps (only a diverged trial is stopped); the verdicts
                 are still computed, and the first step at which each one would have stopped the trial is saved in
                 training_status.json ("would_stop"), to check the stopping rules on complete training curves
Only a converged (or completed) trial is plotted, from the TensorBoard logs: PPO_<env>.png in this directory
(e.g. PPO_Sto1.png), see plot_tb.py.
If no trial converged, no PPO_<env>.png is written (a diagnostic figure <name>_NOT_CONVERGED.png goes to the log
directory instead) and the script exits with code 1. The status of every trial is saved in training_status.json.

CSV data (only for a converged trial, never for the continual learning scripts), in csv_results/ (--csv-dir):
    <tag>_curve.csv       : training curve of the figure (step, team return, smoothed team return)
    <tag>_trajectory.csv  : one deterministic episode of the trained policies (state, action, demand and cost of every
                            day, save_to_csv of the environment), demand seed --csv-seed
    tag = <algo>_<env>_<demand>_lr<lr>_<network>_<steps at convergence>_<date-time>,
    e.g. PPO_Sto1_mu2-std1_lr0.0003_mlp64x64_1.65Msteps_20261011-170501. A new run never overwrites an older file
    (date-time in the name, and a _2, _3 ... suffix if the name already exists).

    python main_ppo.py --env StoSupplyEnv-v3             # std 1 -> PPO_Sto1.png, lr 0.0003
    python main_ppo.py --env StoSupplyEnv-v1 --num-env-runners 4
    python main_ppo.py --lr 0.001 0.0003 0.0001          # grid search over the learning rate (one Tune trial per lr)
    tensorboard --logdir tu_logs                         # watch the training
"""

import argparse
import csv
import datetime
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
from ray import tune

from marl_env import DEFAULT_REWARD_SCALE, ENV_SHORT_NAMES, REWARD_MODES, SupplyChainMARLEnv, build_ppo_config, init_ray
from plot_tb import plot_run, select_trial, smooth

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTTED = ("converged", "completed")   # verdicts of the trials whose figure / CSV are written


def parse_args(default_name="PPO", default_dir="tu"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="StoSupplyEnv-v3",
                        help="environment id (see marl_env.ENV_IDS) or 'module:Class'")
    parser.add_argument("--reward-scale", type=float, default=DEFAULT_REWARD_SCALE,
                        help="rewards are multiplied by this factor (costs are 1e4 - 1e6 per episode)")
    parser.add_argument("--reward-mode", default="team", choices=sorted(REWARD_MODES),
                        help="team: minus the total cost for every agent; individual: minus its own cost; "
                             "mixed: (1 - team_weight) * own + team_weight * total")
    parser.add_argument("--team-weight", type=float, default=0.5, help="weight of the total cost (mixed mode)")
    parser.add_argument("--no-final-costs", dest="final_costs", action="store_false",
                        help="do not charge the costs not settled at the end of the episode (old behaviour)")
    parser.add_argument("--termination-penalty", type=float, default=0.0,
                        help="extra cost of every agent when the episode terminates with too many stockout days")
    parser.add_argument("--max-stockout-days", type=int, default=None,
                        help="terminate the episode after this many stockout days (default: environment value 12)")
    parser.add_argument("--no-stockout-termination", dest="max_stockout_days", action="store_const", const=0,
                        help="never terminate on stockouts: every episode runs the T days")
    parser.add_argument("--lr", type=float, nargs="+", default=[3e-4],
                        help="learning rate; several values -> grid search")
    # 3000 iterations x 3000 timesteps as in the stable-baselines3 version; one timestep is now one decision (reorder day)
    parser.add_argument("--total-timesteps", type=int, default=3000 * 3000)
    parser.add_argument("--time-budget-hours", type=float, default=25.0,
                        help="stop all trials after this time (keep it below the SLURM --time limit)")
    parser.add_argument("--conv-window", type=int, default=50, help="convergence: iterations per moving window")
    parser.add_argument("--conv-tol", type=float, default=0.01,
                        help="convergence: max relative change between two consecutive windows")
    parser.add_argument("--conv-patience", type=int, default=100,
                        help="convergence: iterations the return must stay converged before stopping")
    parser.add_argument("--min-timesteps", type=int, default=500000,
                        help="no convergence / collapse / stall stop before this")
    parser.add_argument("--min-gain", type=float, default=0.05,
                        help="healthy: the return must improve by this fraction since the first window (0: off)")
    parser.add_argument("--near-best-tol", type=float, default=0.05,
                        help="healthy convergence: the last window is within this fraction of the best window")
    parser.add_argument("--collapse-tol", type=float, default=0.2,
                        help="collapse: the last window is more than this fraction below the best window")
    parser.add_argument("--collapse-patience", type=int, default=100,
                        help="collapse: iterations the return must stay collapsed before stopping")
    parser.add_argument("--stall-patience", type=int, default=500,
                        help="stall: iterations without an improvement of --conv-tol before stopping")
    parser.add_argument("--tag", default="",
                        help="suffix of the figure / CSV names, e.g. vfclip -> PPO_Bat10_vfclip.png (variants and test "
                             "runs do not overwrite each other's results)")
    parser.add_argument("--no-early-stop", dest="early_stop", action="store_false",
                        help="run the --total-timesteps: converged / collapsed / stalled verdicts are only recorded")
    parser.add_argument("--vf-clip", type=float, default=10.0,
                        help="clip of the value function squared error (inf: no clipping), see build_ppo_config")
    parser.add_argument("--factory-clip", type=float, default=None,
                        help="PPO clip parameter of the factory policy (default: the common 0.2)")
    parser.add_argument("--csv-dir", default=os.path.join(HERE, "csv_results"),
                        help="CSV data of converged trials (training curve, trajectory of one episode)")
    parser.add_argument("--csv-seed", type=int, default=0, help="demand seed of the episode saved as trajectory CSV")
    parser.add_argument("--checkpoint-freq", type=int, default=50, help="checkpoint every n training iterations")
    parser.add_argument("--num-env-runners", type=int, default=1)
    parser.add_argument("--num-gpus", type=float, default=0, help="GPUs for the learner, e.g. 1 on the cluster")
    parser.add_argument("--name", default=default_name)
    parser.add_argument("--dir", default=default_dir, help="results go to <dir>_logs/<env>/<time>/")
    return parser.parse_args()


def env_config_from_args(args):
    return {"env_class": args.env, "reward_scale": args.reward_scale, "reward_mode": args.reward_mode,
            "team_weight": args.team_weight, "final_costs": args.final_costs,
            "termination_penalty": args.termination_penalty, "max_stockout_days": args.max_stockout_days}


def ppo_kwargs(args):
    """build_ppo_config arguments from the command line (value function clip, clip parameter of the factory)."""
    return {"vf_clip_param": args.vf_clip,
            "clip_params": None if args.factory_clip is None else {"factory": args.factory_clip}}


def lr_search_space(lrs):
    # one value: plain lr; several values: Tune grid search, one trial per lr
    return lrs[0] if len(lrs) == 1 else tune.grid_search(list(lrs))


class TrainingMonitor(tune.Stopper):
    """Checks the reward curve (mean episode return per iteration) of every trial and stops the trial when it has
    converged healthily, or when it is unhealthy (diverged, collapsed, stalled, no learning), see the module
    docstring. The comparisons use the means of `window` iterations; the tolerances are relative to |best window|.
    A trial also stops at max_timesteps, and all trials stop at the time budget, so that the job ends normally
    (checkpoints, figure) before the SLURM time limit. The verdict of each trial is in self.status[trial_id]."""

    def __init__(self, window, tol, patience, min_timesteps, max_timesteps, time_budget_s, min_gain=0.05,
                 near_best_tol=0.05, collapse_tol=0.2, collapse_patience=100, stall_patience=500, early_stop=True):
        self.window, self.tol, self.patience = window, tol, patience
        self.min_timesteps, self.max_timesteps = min_timesteps, max_timesteps
        self.min_gain, self.near_best_tol = min_gain, near_best_tol
        self.collapse_tol, self.collapse_patience, self.stall_patience = collapse_tol, collapse_patience, stall_patience
        self.deadline = time.time() + time_budget_s
        self.history = defaultdict(list)
        self.best = {}                      # best window mean
        self.anchor = {}                    # last best window mean that improved on the previous one by tol
        self.since_best = defaultdict(int)  # iterations since the anchor was set
        self.stable = defaultdict(int)      # consecutive iterations with two consecutive windows within tol
        self.collapsed = defaultdict(int)   # consecutive iterations far below the best
        self.status = {}
        self.early_stop = early_stop
        self.would_stop = defaultdict(dict)  # early_stop False: first step of each verdict

    def _stop(self, trial_id, status, steps, reason):
        self.status[trial_id] = {"status": status, "steps": int(steps), "reason": reason}
        if not self.early_stop:
            self.status[trial_id]["would_stop"] = self.would_stop[trial_id]
        print(f"trial {trial_id} {status} at {steps} steps: {reason}")
        return True

    def _verdict(self, trial_id, status, steps, reason):
        """Converged or unhealthy verdict: stop the trial, or (no early stop) only record its first step."""
        if self.early_stop:
            return self._stop(trial_id, status, steps, reason)
        if status not in self.would_stop[trial_id]:
            self.would_stop[trial_id][status] = {"steps": int(steps), "reason": reason}
            print(f"trial {trial_id} would stop ({status}) at {steps} steps: {reason}")
        return False

    @staticmethod
    def _bad_losses(result):
        """Names of the non-finite learner losses (NaN / inf), e.g. 'factory/total_loss'."""
        bad = []
        for module, metrics in (result.get("learners") or {}).items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if "loss" in k and isinstance(v, (int, float)) and not math.isfinite(v):
                        bad.append(f"{module}/{k}")
        return bad

    def _still_improving(self, h, scale):
        """True if the return still has a significant upward trend (slope > 2 standard errors) that would improve
        it by more than tol within stall_patience iterations (slow improvement hidden in the noise of the windows)."""
        y = np.asarray(h[-(2 * self.window + self.patience):])
        x = np.arange(len(y)) - (len(y) - 1) / 2
        slope = np.dot(x, y - y.mean()) / np.dot(x, x)
        se = np.sqrt(np.sum((y - y.mean() - slope * x) ** 2) / (len(y) - 2) / np.dot(x, x))
        return slope > 2 * se and slope * self.stall_patience > self.tol * scale

    def __call__(self, trial_id, result):
        steps = result.get("num_env_steps_sampled_lifetime",
                           result.get("env_runners", {}).get("num_env_steps_sampled_lifetime", 0))
        bad = self._bad_losses(result)
        if bad:
            return self._stop(trial_id, "diverged", steps, "non-finite loss " + ", ".join(bad))
        ret = result.get("env_runners", {}).get("episode_return_mean")
        if ret is not None and math.isinf(ret):
            return self._stop(trial_id, "diverged", steps, f"episode return {ret}")
        if steps >= self.max_timesteps:
            if not self.early_stop:
                return self._stop(trial_id, "completed", steps, "max timesteps reached (no early stop)")
            return self._stop(trial_id, "max_timesteps", steps, "max timesteps reached before convergence")
        if ret is None or math.isnan(ret):   # no finished episode yet
            return False

        h, w = self.history[trial_id], self.window
        h.append(ret)
        if len(h) < 2 * w:
            return False
        last, prev, first = np.mean(h[-w:]), np.mean(h[-2 * w:-w]), np.mean(h[:w])
        best = self.best[trial_id] = max(self.best.get(trial_id, -np.inf), last)
        scale = max(abs(best), 1e-8)
        anchor = self.anchor.get(trial_id)
        if anchor is None or last > anchor + self.tol * max(abs(anchor), 1e-8):
            self.anchor[trial_id], self.since_best[trial_id] = last, 0
        else:
            self.since_best[trial_id] += 1
        # stable: two consecutive windows differ by less than tol, or than the noise (3 standard errors of the
        # difference of the window means; the returns of a stochastic environment fluctuate more than tol),
        # and no significant upward trend (slow improvement hidden in the noise)
        noise = 3 * np.sqrt((np.var(h[-w:]) + np.var(h[-2 * w:-w])) / w)
        stable = abs(last - prev) <= self.tol * abs(prev) + noise and not self._still_improving(h, scale)
        self.stable[trial_id] = self.stable[trial_id] + 1 if stable else 0
        self.collapsed[trial_id] = self.collapsed[trial_id] + 1 if last < best - self.collapse_tol * scale else 0
        if steps < self.min_timesteps:
            return False

        if self.collapsed[trial_id] >= self.collapse_patience:
            return self._verdict(trial_id, "collapsed", steps,
                              f"return {last:.1f} stayed more than {self.collapse_tol:.0%} below the best {best:.1f}")
        # a stable plateau without enough gain since the start (e.g. the slow start of the training) is not a
        # convergence: the training goes on, and the stall rule stops it if it never improves
        gain = (last - first) / max(abs(first), 1e-8)
        if self.stable[trial_id] >= self.patience and gain >= self.min_gain \
                and last >= best - self.near_best_tol * scale:
            return self._verdict(trial_id, "converged", steps,
                              f"mean episode return {last:.1f} (best {best:.1f}, start {first:.1f})")
        if self.since_best[trial_id] >= self.stall_patience:
            status = "no_learning" if gain < self.min_gain else "stalled"
            return self._verdict(trial_id, status, steps,
                              f"no improvement of {self.tol:.0%} for {self.stall_patience} iterations, return "
                              f"{last:.1f} (best {best:.1f}, start {first:.1f}, gain {gain:+.1%})")
        return False

    def stop_all(self):
        return time.time() >= self.deadline


def fit_env_runners(num_env_runners):
    """A trial needs 1 + num_env_runners CPUs; if the job has fewer, Tune would wait forever (trial PENDING)."""
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    fitted = max(0, min(num_env_runners, cpus - 1))
    if fitted != num_env_runners:
        print(f"WARNING: {cpus} CPUs, num_env_runners reduced from {num_env_runners} to {fitted}")
    return fitted


def env_short_name(env):
    return ENV_SHORT_NAMES.get(env, env.split(":")[-1])


def demand_tag(env_config):
    """Demand setting of the environment for the file names, e.g. mu2-std1 (stochastic), mu2-std0.1-batch10 (batch)."""
    env = SupplyChainMARLEnv(env_config).env
    tag = f"mu{env.muDemand:g}-std{env.stdDemand:g}"
    return tag + (f"-batch{env.batch_size}" if hasattr(env, "batch_size") else "")


def unique_path(path):
    """path, or path with _2, _3 ... before the extension if it exists already (never overwrite an older file)."""
    root, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(path):
        k += 1
        path = f"{root}_{k}{ext}"
    return path


def rollout_episode(checkpoint, env_config, seed=0):
    """One episode of the trained policies (deterministic actions, LSTM state carried for the recurrent policies).
    Returns the wrapped single-planner environment, whose histories hold the whole episode."""
    import torch
    import tree
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
    modules = MultiRLModule.from_checkpoint(os.path.join(checkpoint, "learner_group", "learner", "rl_module"))
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


def save_csv(args, logdir, best, algo, net):
    """CSV data of the converged trial `best`: training curve and the trajectory of one episode, see the docstring."""
    env_config = env_config_from_args(args)
    steps_m = f"{best['steps'] / 1e6:.2f}".rstrip("0").rstrip(".")
    tag = (f"{algo}_{env_short_name(args.env)}{'_' + args.tag if args.tag else ''}_{demand_tag(env_config)}_lr{best['lr']:g}_{net}_{steps_m}Msteps_"
           f"{datetime.datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(args.csv_dir, exist_ok=True)

    _, (steps, returns) = select_trial(logdir, best["lr"])
    curve = unique_path(os.path.join(args.csv_dir, f"{tag}_curve.csv"))
    with open(curve, "x", newline="") as f:
        writer = csv.writer(f)
        # team return: minus the total cost of the supply chain, original cost units (plot_tb.RETURN_TAGS)
        writer.writerow(["step", "team_return", "team_return_smoothed"])
        writer.writerows(zip(steps.astype(int), returns, smooth(returns)))

    trajectory = unique_path(os.path.join(args.csv_dir, f"{tag}_seed{args.csv_seed}_trajectory.csv"))
    env = rollout_episode(best["checkpoint"], env_config, seed=args.csv_seed)
    env.save_to_csv(trajectory)
    # total cost of the episode, same definition as team_return (the Cost column adds up to it, see marl_env.py)
    print(f"csv: {curve}\ncsv: {trajectory} ({env.day} days, total cost {-sum(env.reward_history):.1f})")
    return curve, trajectory


def train(args, config, algo="PPO", net="mlp64x64"):
    # tensorboard logs and checkpoints are saved to this directory
    logdir = os.path.abspath(f"{args.dir}_logs/{env_short_name(args.env)}/{int(time.time())}/")
    os.makedirs(logdir, exist_ok=True)
    stopper = TrainingMonitor(args.conv_window, args.conv_tol, args.conv_patience, args.min_timesteps,
                              args.total_timesteps, args.time_budget_hours * 3600, min_gain=args.min_gain,
                              near_best_tol=args.near_best_tol, collapse_tol=args.collapse_tol,
                              collapse_patience=args.collapse_patience, stall_patience=args.stall_patience,
                              early_stop=args.early_stop)

    init_ray()
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=tune.RunConfig(
            name=args.name,
            storage_path=logdir,
            stop=stopper,
            checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=args.checkpoint_freq,
                                                    checkpoint_at_end=True),
            verbose=1,
        ),
    )
    results = tuner.fit()

    # verdict of every trial; trials still running at the time budget did not converge
    status = []
    for r in results:
        trial_id = (r.metrics or {}).get("trial_id")
        ret = (r.metrics or {}).get("env_runners", {}).get("episode_return_mean")
        s = stopper.status.get(trial_id, {"status": "time_budget", "reason": "time budget reached before convergence"})
        status.append({"trial_id": trial_id, "lr": r.config["lr"], "episode_return_mean": ret, **s,
                       "checkpoint": r.checkpoint.path if r.checkpoint else None})
        print(f"lr: {r.config['lr']}  status: {s['status']}  episode_return_mean: {ret}  ({s['reason']})")
    with open(os.path.join(logdir, "training_status.json"), "w") as f:
        json.dump(status, f, indent=2)

    name = f"{algo}_{env_short_name(args.env)}" + (f"_{args.tag}" if args.tag else "")
    label = f"{algo} in {env_short_name(args.env)}"
    converged = [s for s in status if s["status"] in PLOTTED]
    if not converged:
        # no healthy converged trial: no figure in this directory, only a diagnostic one in the log directory
        best = max(status, key=lambda s: -np.inf if s["episode_return_mean"] is None else s["episode_return_mean"])
        try:
            plot_run(logdir, label=f"{label} (NOT CONVERGED)", out=os.path.join(logdir, f"{name}_NOT_CONVERGED.png"),
                     lr=best["lr"], env=env_short_name(args.env))
        except (FileNotFoundError, ValueError, KeyError) as e:
            print("no diagnostic figure:", e)
        print(f"no trial converged, {name}.png not written; see {logdir}/training_status.json")
        return results, status

    # training curve of the best converged trial, e.g. PPO_Sto1.png in this directory (a copy stays in the log directory)
    best = max(converged, key=lambda s: s["episode_return_mean"])
    print("best lr:", best["lr"], "checkpoint:", best["checkpoint"])
    out = plot_run(logdir, label=label, out=os.path.join(HERE, f"{name}.png"), lr=best["lr"],
                   env=env_short_name(args.env))
    shutil.copy(out, logdir)
    save_csv(args, logdir, best, algo, net)
    return results, status

if __name__ == '__main__':
    args = parse_args(default_name="PPO_pi=[64,64]_vf=[64,64]", default_dir="tu")
    args.num_env_runners = fit_env_runners(args.num_env_runners)
    config = build_ppo_config(env_config_from_args(args), lr=lr_search_space(args.lr), hiddens=(64, 64),
                              num_env_runners=args.num_env_runners, num_gpus=args.num_gpus, **ppo_kwargs(args))
    _, status = train(args, config, algo="PPO")
    sys.exit(0 if any(s["status"] in PLOTTED for s in status) else 1)
