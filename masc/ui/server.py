# -*- coding: utf-8 -*-
"""
masc-ui: local web interface of the masc package (Python standard library only, no extra dependency).

    masc-ui                              # results root = current directory, http://localhost:8765/?token=...
    masc-ui --root runs --port 8800

Pages: training settings (masc-train / masc-train-rppo), continual learning settings (masc-continual), SLURM jobs and
local runs, results (training curves, figures, re-plotting) and an environment explorer (fixed policies).
The forms are built from the command lines of masc.train / masc.continual (build_parser), so a new option appears in
the interface without changing it, and every command is validated by the same argparse parser before it is run.

Jobs: a SLURM job is an sbatch script generated in <root>/ui_jobs/ (one array task per environment / continual task),
which runs `python -m masc.train ...` with the Python of this server; it is written to the results root, as the
commands themselves. A local run starts the same command in the background (not on a login node: use SLURM there).

Security: the server listens on 127.0.0.1 only (other users of a shared login node can reach it, so every API call
needs the random token printed at the start). Open it from your own computer through an SSH tunnel:
    ssh -N -L 8765:localhost:8765 <user>@<login node where masc-ui runs>
"""

import argparse
import datetime
import glob
import json
import math
import mimetypes
import os
import re
import secrets
import shlex
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import __version__
from .. import continual as continual_mod
from .. import train as train_mod
from ..marl_env import ENV_IDS, ENV_SHORT_NAMES, SupplyChainMARLEnv

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
AGENTS = ("factory", "warehouse", "retailer")

# groups of the form fields (argparse dest -> group), the others go to "其他"
GROUPS = {
    "reward_mode": "环境与奖励", "reward_scale": "环境与奖励", "team_weight": "环境与奖励", "final_costs": "环境与奖励",
    "termination_penalty": "环境与奖励", "max_stockout_days": "环境与奖励",
    "lr": "训练", "seeds": "训练", "total_timesteps": "训练", "early_stop": "训练", "time_budget_hours": "训练",
    "num_env_runners": "训练", "num_gpus": "训练", "checkpoint_freq": "训练", "rounds": "训练",
    "timesteps_per_phase": "训练",
    "vf_clip": "PPO", "factory_clip": "PPO",
    "conv_window": "收敛判据", "conv_tol": "收敛判据", "conv_patience": "收敛判据", "min_timesteps": "收敛判据",
    "min_gain": "收敛判据", "near_best_tol": "收敛判据", "collapse_tol": "收敛判据", "collapse_patience": "收敛判据",
    "stall_patience": "收敛判据", "no_monitor": "收敛判据", "monitor_window": "收敛判据", "no_learning_steps": "收敛判据",
    "out_dir": "输出", "csv_dir": "输出", "csv_seed": "输出", "name": "输出", "dir": "输出", "tag": "输出",
}
# commands of the interface: module run with "python -m", its parser, and the option of the item of an array task
COMMANDS = {
    "train": {"module": "masc.train", "parser": lambda: train_mod.build_parser("PPO_pi=[64,64]_vf=[64,64]", "tu"),
              "item_option": "--env", "items": list(ENV_IDS)},
    "train_rppo": {"module": "masc.train_rppo",
                   "parser": lambda: train_mod.build_parser("RecurrentPPO_pi=64,64_vf=64,64_lstm=1", "tu1"),
                   "item_option": "--env", "items": list(ENV_IDS)},
    "continual": {"module": "masc.continual", "parser": continual_mod.build_parser,
                  "item_option": "--task", "items": list(continual_mod.TASKS)},
}
# options handled by the interface itself (environment / task of the array task, output directory = results root)
HIDDEN = {"env", "task", "out_dir", "help"}
ID_RE = re.compile(r"^\d+(_\d+)?$")
SLURM_FIELD_RE = re.compile(r"^[A-Za-z0-9_.:,@+-]*$")


class UIError(Exception):
    """Error shown to the user (HTTP 400)."""


# ----------------------------------------------------------------------------------------------------------------
# forms and command lines
# ----------------------------------------------------------------------------------------------------------------

def _json_default(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def parser_schema(parser):
    """Form fields of an argparse parser: [{option, dest, kind (flag / value), type, multiple, choices, default,
    help, group}]. A flag is emitted when checked (store_true / store_false / store_const options)."""
    fields = []
    for a in parser._actions:
        if a.dest in HIDDEN or not a.option_strings:
            continue
        option = max(a.option_strings, key=len)
        flag = isinstance(a, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._StoreConstAction))
        field = {"option": option, "dest": a.dest, "kind": "flag" if flag else "value", "help": a.help or "",
                 "group": GROUPS.get(a.dest, "其他")}
        if not flag:
            default = a.default
            if isinstance(default, (list, tuple)):
                default = " ".join(str(_json_default(v)) for v in default)
            field.update(type=a.type.__name__ if a.type else "str", multiple=a.nargs in ("+", "*"),
                         choices=list(a.choices) if a.choices else None,
                         default=_json_default(default) if default is not None else None)
        fields.append(field)
    return fields


def _raising_parser(kind):
    parser = COMMANDS[kind]["parser"]()

    def error(message):
        raise UIError(message)
    parser.error = error
    return parser


def build_argv(kind, values):
    """Command-line arguments from the form values {option: value} (only the values that differ from the defaults),
    validated by the argparse parser of the command. Returns (argv, parsed namespace of the first item)."""
    if kind not in COMMANDS:
        raise UIError(f"unknown command {kind}")
    fields = {f["option"]: f for f in parser_schema(COMMANDS[kind]["parser"]())}
    argv = []
    for option, value in (values or {}).items():
        field = fields.get(option)
        if field is None:
            raise UIError(f"unknown option {option}")
        if field["kind"] == "flag":
            if value is True:
                argv.append(option)
            continue
        text = str(value).strip() if value is not None else ""
        if not text:
            continue
        tokens = [t for t in re.split(r"[\s,]+", text) if t] if field["multiple"] else [text]
        default = field["default"]
        default_tokens = [] if default is None else ([t for t in str(default).split()] if field["multiple"]
                                                     else [str(default)])
        if tokens == default_tokens:
            continue
        argv += [option] + tokens
    item = COMMANDS[kind]["items"][0]
    try:
        namespace = _raising_parser(kind).parse_args([COMMANDS[kind]["item_option"], item] + argv)
    except UIError as e:
        raise UIError(f"参数错误: {e}")
    return argv, namespace


def python_command(kind, item, argv):
    return [sys.executable, "-m", COMMANDS[kind]["module"], COMMANDS[kind]["item_option"], item] + list(argv)


def sbatch_script(kind, items, argv, root, slurm):
    """sbatch script: one array task per item (environment / continual task), random start delay, one retry if the
    command fails within 5 minutes (the local Ray instance did not start, see jobs/job_train.sh)."""
    for key in ("job_name", "partition", "cpus", "mem", "time", "mail"):
        if not SLURM_FIELD_RE.match(str(slurm.get(key, ""))):
            raise UIError(f"SLURM 设置 {key} 含有不允许的字符")
    mail = (f"#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user={slurm['mail']}\n" if slurm.get("mail") else "")
    command = " ".join(shlex.quote(t) for t in python_command(kind, "ITEM_PLACEHOLDER", argv))
    command = command.replace("ITEM_PLACEHOLDER", '"$ITEM"')
    return f"""#!/bin/bash
#SBATCH --job-name={slurm['job_name']}
#SBATCH --partition={slurm['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm['cpus']}
#SBATCH --mem={slurm['mem']}
#SBATCH --time={slurm['time']}
#SBATCH --array=0-{len(items) - 1}
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
{mail}
# generated by masc-ui {__version__} on {datetime.datetime.now():%Y-%m-%d %H:%M:%S}
set -o pipefail
cd {shlex.quote(root)}
export PYTHONUNBUFFERED=1
ITEMS=({" ".join(shlex.quote(i) for i in items)})
ITEM=${{ITEMS[${{SLURM_ARRAY_TASK_ID:-0}}]}}
echo "job ${{SLURM_JOB_ID:-local}} task ${{SLURM_ARRAY_TASK_ID:-0}} on $(hostname): $ITEM"
sleep $((RANDOM % 60))   # many jobs can start on the same node at the same time
for attempt in 1 2; do
    start=$(date +%s)
    status=0
    {command} || status=$?
    if [ "$status" = 0 ] || [ $(( $(date +%s) - start )) -gt 300 ] || [ "$attempt" = 2 ]; then
        exit "$status"
    fi
    echo "failed after $(( $(date +%s) - start )) s (exit $status), retrying in 60 s"
    sleep 60
done
"""


# ----------------------------------------------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------------------------------------------

def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def find_runs(root, max_depth=4):
    """Training runs (<x>_logs/<env>/<time>/) and continual learning runs (PPOcon*_models/<task>/<stamp>/lr=<lr>/)
    under root, newest first."""
    runs = []
    root = os.path.abspath(root)
    for dirpath, dirnames, _ in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        # do not descend into trial / checkpoint directories
        dirnames[:] = [d for d in dirnames if not d.startswith(("PPO_", "RecurrentPPO", "checkpoint_", ".", "ray_"))
                       and d not in ("tb", "__pycache__", "csv_results")]
        if depth >= max_depth:
            dirnames[:] = []
        name = os.path.basename(dirpath)
        if name.endswith("_logs"):
            for env in sorted(os.listdir(dirpath)):
                for stamp in sorted(glob.glob(os.path.join(glob.escape(os.path.join(dirpath, env)), "*"))):
                    if os.path.isdir(stamp):
                        runs.append(_training_run(root, stamp, env))
            dirnames[:] = []
        elif name.startswith("PPOcon") and name.endswith("_models"):
            for status_dir in sorted(glob.glob(os.path.join(glob.escape(dirpath), "*", "*", "lr=*"))):
                runs.append(_continual_run(root, status_dir))
            dirnames[:] = []
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def _trial_dirs(run_dir):
    files = glob.glob(os.path.join(glob.escape(run_dir), "*", "*", "progress.csv"))
    return sorted(os.path.dirname(f) for f in files)


def _training_run(root, run_dir, env):
    status = _read_json(os.path.join(run_dir, "training_status.json"))
    trials = _trial_dirs(run_dir)
    mtime = max([os.path.getmtime(os.path.join(t, "progress.csv")) for t in trials] + [os.path.getmtime(run_dir)])
    if status:
        states = sorted({s.get("status") for s in status})
        state = ", ".join(states)
    else:
        state = "运行中" if time.time() - mtime < 600 else "未完成"
    return {"type": "training", "env": env, "path": os.path.relpath(run_dir, root), "time": _stamp(run_dir),
            "mtime": mtime, "state": state, "n_trials": len(trials), "status": status}


def _continual_run(root, lr_dir):
    status = _read_json(os.path.join(lr_dir, "continual_status.json"))
    task = os.path.basename(os.path.dirname(os.path.dirname(lr_dir)))
    tb = os.path.join(lr_dir, "tb")
    mtime = max([os.path.getmtime(f) for f in glob.glob(os.path.join(glob.escape(tb), "*"))] +
                [os.path.getmtime(lr_dir)])
    state = status.get("status") if status else ("运行中" if time.time() - mtime < 600 else "未完成")
    return {"type": "continual", "env": task, "path": os.path.relpath(lr_dir, root),
            "time": os.path.basename(os.path.dirname(lr_dir)), "mtime": mtime, "state": state, "n_trials": 1,
            "status": status}


def _stamp(run_dir):
    name = os.path.basename(run_dir)
    if name.isdigit():
        return datetime.datetime.fromtimestamp(int(name)).strftime("%Y-%m-%d %H:%M")
    return name


def _downsample(x, y, n=800):
    if len(x) <= n:
        return x, y
    idx = np.linspace(0, len(x) - 1, n).round().astype(int)
    return [x[i] for i in idx], [y[i] for i in idx]


def run_curves(root, rel_path):
    """Curves of a run: training: team_return of every trial (and the mean +- std of the trials of the same lr if
    they have several seeds); continual: episode_return_mean of every phase."""
    import pandas as pd
    run_dir = safe_path(root, rel_path)
    if os.path.isdir(os.path.join(run_dir, "tb")):     # continual learning run
        from ..plotting import read_phases
        phases = read_phases(os.path.join(run_dir, "tb"))
        series = []
        for number, env, steps, returns in phases:
            x, y = _downsample(list(steps), [float(v) for v in returns])
            series.append({"name": f"phase {number} {env}", "env": env, "x": x, "y": y})
        return {"type": "continual", "series": series, "ylabel": "episode_return_mean (scaled)"}
    series, groups = [], {}
    for trial in _trial_dirs(run_dir):
        try:
            d = pd.read_csv(os.path.join(trial, "progress.csv"))
        except (OSError, ValueError):
            continue
        col = "env_runners/team_return" if "env_runners/team_return" in d else "env_runners/episode_return_mean"
        if col not in d or "num_env_steps_sampled_lifetime" not in d:
            continue
        d = d[["num_env_steps_sampled_lifetime", col]].dropna()
        params = _read_json(os.path.join(trial, "params.json")) or {}
        lr, seed = params.get("lr"), params.get("seed")
        x, y = d["num_env_steps_sampled_lifetime"].astype(int).tolist(), d[col].astype(float).tolist()
        groups.setdefault(lr, []).append(dict(zip(x, y)))
        xs, ys = _downsample(x, y)
        series.append({"name": f"lr={lr}" + (f" seed={seed}" if seed is not None else ""), "lr": lr, "seed": seed,
                       "x": xs, "y": ys, "last": y[-1] if y else None, "iterations": len(y)})
    bands = []
    for lr, curves in groups.items():
        if len(curves) < 2:
            continue
        common = sorted(set.intersection(*(set(c) for c in curves)))
        if not common:
            continue
        values = np.array([[c[s] for s in common] for c in curves])
        xs, mean = _downsample(common, values.mean(axis=0).tolist())
        _, std = _downsample(common, values.std(axis=0).tolist())
        bands.append({"name": f"lr={lr} 均值 ± std（{len(curves)} 个种子）", "lr": lr, "x": xs, "y": mean,
                      "lo": [m - s for m, s in zip(mean, std)], "hi": [m + s for m, s in zip(mean, std)]})
    return {"type": "training", "series": series, "bands": bands, "ylabel": "team_return（episode 总成本，取负）"}


def run_figures(root, rel_path):
    run_dir = safe_path(root, rel_path)
    files = sorted(glob.glob(os.path.join(glob.escape(run_dir), "*.png")))
    return [os.path.relpath(f, root) for f in files]


def safe_path(root, rel_path):
    """Absolute path of rel_path inside root (no access outside the results root)."""
    root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(root, rel_path or ""))
    if path != root and not path.startswith(root + os.sep):
        raise UIError("路径不在结果目录内")
    return path


# ----------------------------------------------------------------------------------------------------------------
# environment explorer
# ----------------------------------------------------------------------------------------------------------------

def simulate(env_id, orders, reorder_point, n_seeds=10, max_stockout_days=0, final_costs=True, first_seed=100):
    """Fixed policy (order quantities of the three agents, reorder point of the retailer) on n_seeds episodes:
    mean / std of the episode total cost (team_return), mean cost of every agent, trace of the first episode."""
    if env_id not in ENV_IDS:
        raise UIError(f"unknown environment {env_id}")
    n_seeds = int(min(max(n_seeds, 1), 200))
    policy = {a: np.array([0, int(np.clip(orders[a], 0, 19)), 0]) for a in AGENTS}
    policy["retailer"][2] = int(np.clip(reorder_point, 0, 9))
    config = {"env_class": env_id, "max_stockout_days": int(max_stockout_days), "final_costs": bool(final_costs),
              "reward_mode": "individual", "reward_scale": 1.0}
    totals, per_agent, trace = [], np.zeros(3), None
    for k in range(n_seeds):
        env = SupplyChainMARLEnv(config)
        _, info = env.reset(seed=first_seed + k)
        total, own, done = info["factory"]["team_reward"], np.zeros(3), False
        while not done:
            _, rewards, terminated, truncated, info = env.step(policy)
            total += info["factory"]["team_reward"]
            own += [rewards[a] for a in AGENTS]
            done = terminated["__all__"] or truncated["__all__"]
        totals.append(total)
        per_agent += own
        if k == 0:
            inner = env.env
            states = np.array(inner.state_history)
            trace = {"day": list(range(len(states))),
                     "inventory": {a: states[:, 0, i].astype(float).tolist() for i, a in enumerate(AGENTS)},
                     "demand": [0.0] + [float(d) for d in inner.demand_history],
                     "cost": [0.0] + [-float(c) for c in inner.reward_history],
                     "order_day": [False] + [bool(t) for t in inner.actionTriggerHistory],
                     "days": int(inner.day), "terminated": bool(terminated["__all__"])}
    return {"mean": float(np.mean(totals)), "std": float(np.std(totals)), "n": n_seeds,
            "per_agent": dict(zip(AGENTS, (per_agent / n_seeds).tolist())), "trace": trace}


def best_fixed_policies(env_id, n_seeds=5, max_stockout_days=0, top=10):
    """Grid search over fixed policies: retailer order q (2-19) and reorder point (0-9), upstream order q or 19."""
    results = []
    for q in range(2, 20):
        for rp in range(10):
            for up in sorted({q, 19}):
                r = simulate(env_id, {"factory": up, "warehouse": up, "retailer": q}, rp, n_seeds,
                             max_stockout_days, first_seed=0)
                results.append({"q_retailer": q, "reorder_point": rp, "q_upstream": up, "mean": r["mean"]})
    results.sort(key=lambda r: r["mean"], reverse=True)
    return results[:top]


# ----------------------------------------------------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------------------------------------------------

class App:
    def __init__(self, root, token):
        self.root = os.path.abspath(root)
        self.token = token
        self.local_runs = {}             # id -> {"items", "commands", "log", "status", "process"}
        self.lock = threading.Lock()
        self.plot_lock = threading.Lock()
        self.has_slurm = shutil_which("sbatch") is not None
        self.hostname = socket.gethostname()

    # ----- meta -----
    def meta(self):
        from ..plotting import ENV_COLORS, XLIM, YLIM
        return {
            "version": __version__, "root": self.root, "hostname": self.hostname, "python": sys.executable,
            "has_slurm": self.has_slurm, "login_node": "login" in self.hostname,
            "user": os.environ.get("USER", ""),
            "envs": [{"id": k, "short": ENV_SHORT_NAMES[k], "color": ENV_COLORS.get(ENV_SHORT_NAMES[k], "gray"),
                      "cls": ENV_IDS[k]} for k in ENV_IDS],
            "tasks": [{"id": t, "envs": [ENV_SHORT_NAMES[e] for e in envs], "rounds": continual_mod.ROUNDS[t]}
                      for t, envs in continual_mod.TASKS.items()],
            "colors": ENV_COLORS, "xlim": XLIM, "ylim": YLIM,
            "schemas": {kind: parser_schema(c["parser"]()) for kind, c in COMMANDS.items()},
        }

    # ----- commands -----
    def preview(self, body):
        kind, items = body.get("kind"), body.get("items") or []
        argv, _ = build_argv(kind, body.get("values"))
        if not items:
            raise UIError("请至少选择一个环境 / 任务")
        bad = [i for i in items if i not in COMMANDS[kind]["items"]]
        if bad:
            raise UIError(f"unknown items {bad}")
        commands = [" ".join(shlex.quote(t) for t in python_command(kind, i, argv)) for i in items]
        script = sbatch_script(kind, items, argv, self.root, body.get("slurm") or default_slurm(kind))
        return {"argv": argv, "commands": commands, "script": script}

    def submit(self, body):
        p = self.preview(body)
        kind, items = body["kind"], body["items"]
        if body.get("target") == "local":
            return self._run_local(kind, items, p["argv"])
        if not self.has_slurm:
            raise UIError("没有找到 sbatch：这台机器上没有 SLURM，请选择本地运行")
        jobs_dir = os.path.join(self.root, "ui_jobs")
        os.makedirs(jobs_dir, exist_ok=True)
        slurm = body.get("slurm") or default_slurm(kind)
        script_path = os.path.join(jobs_dir, f"{datetime.datetime.now():%Y%m%d-%H%M%S}_{slurm['job_name']}.sh")
        with open(script_path, "w") as f:
            f.write(p["script"])
        out = subprocess.run(["sbatch", script_path], cwd=self.root, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise UIError(f"sbatch 失败: {out.stderr.strip() or out.stdout.strip()}")
        return {"message": out.stdout.strip(), "script": os.path.relpath(script_path, self.root)}

    def _run_local(self, kind, items, argv):
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.root, "ui_local")
        os.makedirs(log_dir, exist_ok=True)
        log = os.path.join(log_dir, f"{run_id}_{kind}.log")
        commands = [python_command(kind, i, argv) for i in items]
        run = {"id": run_id, "kind": kind, "items": items, "log": os.path.relpath(log, self.root),
               "status": "运行中", "process": None, "started": time.time()}
        with self.lock:
            self.local_runs[run_id] = run

        def worker():
            with open(log, "a") as f:
                for cmd in commands:
                    if run["status"] == "已取消":
                        break
                    f.write(f"\n$ {' '.join(shlex.quote(t) for t in cmd)}\n")
                    f.flush()
                    run["process"] = subprocess.Popen(cmd, cwd=self.root, stdout=f, stderr=subprocess.STDOUT,
                                                      env={**os.environ, "PYTHONUNBUFFERED": "1"})
                    code = run["process"].wait()
                    f.write(f"[exit {code}]\n")
            if run["status"] != "已取消":
                run["status"] = "已结束"
        threading.Thread(target=worker, daemon=True).start()
        return {"message": f"本地运行 {run_id} 已开始（{len(items)} 个命令依次运行）", "log": run["log"]}

    # ----- jobs -----
    def jobs(self):
        result = {"squeue": [], "sacct": [], "local": [], "has_slurm": self.has_slurm}
        user = os.environ.get("USER", "")
        if self.has_slurm:
            out = _run(["squeue", "-u", user, "-h", "-o", "%i|%j|%T|%M|%l|%R|%Z"])
            for line in out.splitlines():
                parts = line.split("|")
                if len(parts) == 7:
                    result["squeue"].append(dict(zip(["id", "name", "state", "time", "limit", "reason", "workdir"],
                                                     parts)))
            out = _run(["sacct", "-u", user, "-X", "-n", "-P", "-S", "now-2days",
                        "-o", "JobID,JobName,State,Elapsed,End,WorkDir"])
            for line in out.splitlines()[-200:]:
                parts = line.split("|")
                if len(parts) == 6 and parts[2] not in ("RUNNING", "PENDING"):
                    result["sacct"].append(dict(zip(["id", "name", "state", "elapsed", "end", "workdir"], parts)))
            result["sacct"].reverse()
        with self.lock:
            for run in self.local_runs.values():     # the status is updated by the worker thread of the run
                result["local"].append({k: run[k] for k in ("id", "kind", "items", "log", "status")})
        return result

    def job_logs(self, job_id):
        """Output files of a SLURM job (array task id like 123_4), found by scontrol (running jobs) or by the
        %x_%A_%a.out / %x_%j.out patterns in its working directory (finished jobs)."""
        if not ID_RE.match(job_id or ""):
            raise UIError("作业号格式不对")
        out = _run(["scontrol", "show", "job", job_id])
        files = re.findall(r"Std(?:Out|Err)=(\S+)", out)
        if not files:
            info = _run(["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "JobName,WorkDir"]).splitlines()
            if info:
                name, workdir = info[0].split("|", 1)
                array_id, _, task = job_id.partition("_")
                pattern = f"{name}_{array_id}_{task}.*" if task else f"{name}_{array_id}.*"
                files = sorted(glob.glob(os.path.join(glob.escape(workdir), pattern)))
        return [f for f in files if os.path.isfile(f)]

    def cancel(self, body):
        job_id = str(body.get("id", ""))
        if body.get("local"):
            with self.lock:
                run = self.local_runs.get(job_id)
            if run is None:
                raise UIError("没有这个本地运行")
            run["status"] = "已取消"
            if run["process"] is not None and run["process"].poll() is None:
                run["process"].terminate()
            return {"message": f"本地运行 {job_id} 已取消"}
        if not ID_RE.match(job_id):
            raise UIError("作业号格式不对")
        out = subprocess.run(["scancel", job_id], capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            raise UIError(f"scancel 失败: {out.stderr.strip()}")
        return {"message": f"已取消作业 {job_id}"}

    def read_log(self, path, lines=200):
        """Last lines of a log file: inside the results root, or an output file of one of the user's SLURM jobs."""
        real = os.path.realpath(path if os.path.isabs(path) else os.path.join(self.root, path))
        root = os.path.realpath(self.root)
        allowed = real.startswith(root + os.sep) or (os.path.isabs(path) and real.endswith((".out", ".err"))
                                                     and _owned_by_user(real))
        if not allowed or not os.path.isfile(real):
            raise UIError("不能读取这个文件")
        with open(real, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 400_000))
            text = f.read().decode("utf-8", "replace")
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        return {"path": real, "text": "\n".join(text.splitlines()[-int(lines):]), "size": size}

    # ----- plots -----
    def replot(self, body):
        from ..plotting import plot_run, plot_seeds
        run_dir = safe_path(self.root, body.get("path"))
        out_dir = os.path.join(self.root, "ui_plots")
        os.makedirs(out_dir, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", body.get("out") or "plot") + ".png"
        out = os.path.join(out_dir, name)
        xlim = tuple(float(v) for v in body.get("xlim"))
        ylim = tuple(float(v) for v in body.get("ylim"))
        lr = float(body["lr"]) if body.get("lr") not in (None, "") else None
        with self.plot_lock:
            if body.get("seeds"):
                plot_seeds(run_dir, body.get("label") or "PPO", out, lr=lr, xlim=xlim, ylim=ylim,
                           smoothing=float(body.get("smoothing") or 0))
            else:
                plot_run(run_dir, body.get("label") or "PPO", out, lr=lr, xlim=xlim, ylim=ylim,
                         smoothing=float(body.get("smoothing") if body.get("smoothing") not in (None, "") else 0.6))
        return {"path": os.path.relpath(out, self.root), "time": time.time()}


def default_slurm(kind):
    return {"job_name": f"masc_{kind}", "partition": "batch", "cpus": 16, "mem": "20G", "time": "26:00:00", "mail": ""}


def _owned_by_user(path):
    try:
        return os.stat(path).st_uid == os.getuid()
    except OSError:
        return False


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def shutil_which(name):
    import shutil
    return shutil.which(name)


# ----------------------------------------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------------------------------------

def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "masc-ui"

        def log_message(self, fmt, *args):     # quiet: no line per request
            pass

        def _send(self, code, body, content_type="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=_json_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self, query):
            token = self.headers.get("X-Token") or (query.get("token") or [""])[0]
            return secrets.compare_digest(token, app.token)

        def _handle(self, method):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if not url.path.startswith("/api/"):
                return self._send(404, {"error": "not found"})
            if not self._authorised(query):
                return self._send(403, {"error": "令牌无效：请使用 masc-ui 启动时打印的完整网址（含 ?token=...）"})
            q = {k: v[0] for k, v in query.items()}
            try:
                body = {}
                if method == "POST":
                    length = int(self.headers.get("Content-Length") or 0)
                    try:
                        body = json.loads(self.rfile.read(length) or b"{}")
                    except ValueError:
                        raise UIError("请求不是合法的 JSON")
                route = url.path[len("/api/"):]
                if route == "file":
                    path = safe_path(app.root, q.get("path"))
                    if not os.path.isfile(path):
                        raise UIError("文件不存在")
                    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
                    with open(path, "rb") as f:
                        return self._send(200, f.read(), ctype)
                handlers = {
                    ("GET", "meta"): app.meta,
                    ("POST", "preview"): lambda: app.preview(body),
                    ("POST", "submit"): lambda: app.submit(body),
                    ("GET", "jobs"): app.jobs,
                    ("GET", "job_logs"): lambda: {"files": app.job_logs(q.get("id"))},
                    ("GET", "log"): lambda: app.read_log(q.get("path", ""), int(q.get("lines", 200))),
                    ("POST", "cancel"): lambda: app.cancel(body),
                    ("GET", "runs"): lambda: {"runs": find_runs(app.root)},
                    ("GET", "curves"): lambda: run_curves(app.root, q.get("path")),
                    ("GET", "figures"): lambda: {"figures": run_figures(app.root, q.get("path"))},
                    ("POST", "plot"): lambda: app.replot(body),
                    ("POST", "simulate"): lambda: simulate(
                        body.get("env"), body.get("orders"), body.get("reorder_point", 6), body.get("seeds", 10),
                        body.get("max_stockout_days", 0), body.get("final_costs", True)),
                    ("POST", "best_fixed"): lambda: {"best": best_fixed_policies(
                        body.get("env"), int(body.get("seeds", 5)), int(body.get("max_stockout_days", 0)))},
                }
                fn = handlers.get((method, route))
                if fn is None:
                    return self._send(404, {"error": f"unknown API {method} {route}"})
                return self._send(200, fn())
            except UIError as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:                       # shown in the page, the server keeps running
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


def serve(root=".", host="127.0.0.1", port=8765, token=None, open_message=True):
    """Start the server (first free port from `port`); returns (server, app, url)."""
    app = App(root, token or secrets.token_urlsafe(16))
    handler = make_handler(app)
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, p), handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port in {port}-{port + 19}")
    port = server.server_address[1]
    url = f"http://localhost:{port}/?token={app.token}"
    if open_message:
        user = os.environ.get("USER", "<user>")
        print(f"masc-ui {__version__} on {app.hostname}, results root {app.root}")
        print(f"  open:  {url}")
        if host in ("127.0.0.1", "localhost"):
            print(f"  from your computer: ssh -N -L {port}:localhost:{port} {user}@{socket.getfqdn()}")
            print("  (the tunnel must go to this login node; then open the address above in the browser)")
        print("  stop with Ctrl-C", flush=True)
    return server, app, url


def main(argv=None):
    parser = argparse.ArgumentParser(prog="masc-ui", description="local web interface of the masc package")
    parser.add_argument("--root", default=".", help="results directory: commands run and write there, its runs are "
                                                     "shown (default: current directory)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1",
                        help="listen address (default 127.0.0.1; other addresses expose the interface to the network)")
    args = parser.parse_args(argv)
    os.makedirs(args.root, exist_ok=True)
    server, _, _ = serve(args.root, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
