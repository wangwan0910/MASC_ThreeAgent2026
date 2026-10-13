# -*- coding: utf-8 -*-
"""
masc-ui: local web interface of the masc package, for a Mac (Python standard library only, no extra dependency).
It is not connected to the Iridis cluster: on Iridis, use the commands (masc-train ...) and the job scripts of jobs/.

    masc-ui                    # results root = current directory; opens http://localhost:8765/?token=... in the browser
    masc-ui --root runs --port 8800 --no-browser

Pages: training settings (masc-train / masc-train-transformer), continual learning settings (masc-continual), local runs,
results (training curves, figures, re-plotting; also results copied from Iridis with rsync) and an environment
explorer (fixed policies).
The forms are built from the command lines of masc.train / masc.continual (build_parser), so a new option appears in
the interface without changing it, and every command is validated by the same argparse parser before it is run.

Runs: the commands (`python -m masc.train ...` with the Python of this server) run on this computer, in the results
root, one after the other (a queue: several Ray trainings at the same time would compete for the CPUs of a laptop).
Their output goes to <root>/ui_local/.

Security: the server listens on 127.0.0.1 only, and every API call needs the random token of the address opened at
the start (another web page open in the browser cannot start commands through it).
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
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import __version__
from .. import continual as continual_mod
from .. import main_transformerPPO as transformer_mod
from .. import train as train_mod
from ..marl_env import ENV_IDS, ENV_SHORT_NAMES, SupplyChainMARLEnv

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
AGENTS = ("factory", "warehouse", "retailer")

# groups of the form fields (argparse dest -> group), the others go to "Other"
GROUPS = {
    "reward_mode": "Environment and reward", "reward_scale": "Environment and reward", "team_weight": "Environment and reward", "final_costs": "Environment and reward",
    "termination_penalty": "Environment and reward", "max_stockout_days": "Environment and reward",
    "lr": "Training", "seeds": "Training", "total_timesteps": "Training", "early_stop": "Training", "time_budget_hours": "Training",
    "num_env_runners": "Training", "num_gpus": "Training", "checkpoint_freq": "Training", "rounds": "Training",
    "timesteps_per_phase": "Training",
    "vf_clip": "PPO", "factory_clip": "PPO",
    **{f"tf_{key}": "Transformer" for key in ("memory", "num_layers", "d_model", "num_heads", "head_dim", "ff_dim",
                                              "gate_bias", "max_seq_len", "head_hiddens")},
    "conv_window": "Convergence criteria", "conv_tol": "Convergence criteria", "conv_patience": "Convergence criteria", "min_timesteps": "Convergence criteria",
    "min_gain": "Convergence criteria", "near_best_tol": "Convergence criteria", "collapse_tol": "Convergence criteria", "collapse_patience": "Convergence criteria",
    "stall_patience": "Convergence criteria", "no_monitor": "Convergence criteria", "monitor_window": "Convergence criteria", "no_learning_steps": "Convergence criteria",
    "out_dir": "Output", "csv_dir": "Output", "csv_seed": "Output", "name": "Output", "dir": "Output", "tag": "Output",
}
# commands of the interface: module run with "python -m", its parser, and the option of the item (environment / task)
COMMANDS = {
    "train": {"module": "masc.train", "parser": lambda: train_mod.build_parser("PPO_pi=[64,64]_vf=[64,64]", "tu"),
              "item_option": "--env", "items": list(ENV_IDS)},
    "train_transformer": {"module": "masc.main_transformerPPO", "parser": transformer_mod.build_parser,
                          "item_option": "--env", "items": list(ENV_IDS)},
    "continual": {"module": "masc.continual", "parser": continual_mod.build_parser,
                  "item_option": "--task", "items": list(continual_mod.TASKS)},
}
# algorithms of the training page (commands above), in the order of the page
ALGORITHMS = {"train": "PPO (MLP)", "train_transformer": "Transformer PPO (GTrXL)"}
# options handled by the interface itself (environment / task, output directory = results root)
HIDDEN = {"env", "task", "out_dir", "help"}


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
                 "group": GROUPS.get(a.dest, "Other")}
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
        raise UIError(f"Invalid arguments: {e}")
    return argv, namespace


def python_command(kind, item, argv):
    return [sys.executable, "-m", COMMANDS[kind]["module"], COMMANDS[kind]["item_option"], item] + list(argv)


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
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(("PPO_", "TransformerPPO", "TPPO_", "checkpoint_", ".", "ray_"))
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
        state = "running" if time.time() - mtime < 600 else "unfinished"
    return {"type": "training", "env": env, "path": os.path.relpath(run_dir, root), "time": _stamp(run_dir),
            "mtime": mtime, "state": state, "n_trials": len(trials), "status": status}


def _continual_run(root, lr_dir):
    status = _read_json(os.path.join(lr_dir, "continual_status.json"))
    task = os.path.basename(os.path.dirname(os.path.dirname(lr_dir)))
    tb = os.path.join(lr_dir, "tb")
    mtime = max([os.path.getmtime(f) for f in glob.glob(os.path.join(glob.escape(tb), "*"))] +
                [os.path.getmtime(lr_dir)])
    state = status.get("status") if status else ("running" if time.time() - mtime < 600 else "unfinished")
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
    from ..plotting import align_curves
    run_dir = safe_path(root, rel_path)
    if os.path.isdir(os.path.join(run_dir, "tb")):     # continual learning run
        from ..plotting import METRIC_LABELS, read_phases_metric
        phases, metric = read_phases_metric(os.path.join(run_dir, "tb"))
        series = []
        for number, env, steps, returns in phases:
            x, y = _downsample(list(steps), [float(v) for v in returns])
            series.append({"name": f"phase {number} {env}", "env": env, "x": x, "y": y})
        return {"type": "continual", "series": series, "ylabel": METRIC_LABELS[metric]}
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
        grid, values = align_curves([(sorted(c), [c[k] for k in sorted(c)]) for c in curves])
        if not len(grid):
            continue
        xs, mean = _downsample(grid.tolist(), values.mean(axis=0).tolist())
        _, std = _downsample(grid.tolist(), values.std(axis=0).tolist())
        bands.append({"name": f"lr={lr} mean ± std ({len(curves)} seeds)", "lr": lr, "x": xs, "y": mean,
                      "lo": [m - s for m, s in zip(mean, std)], "hi": [m + s for m, s in zip(mean, std)]})
    return {"type": "training", "series": series, "bands": bands, "ylabel": "team_return (negative total episode cost)"}


def run_figures(root, rel_path):
    run_dir = safe_path(root, rel_path)
    files = sorted(glob.glob(os.path.join(glob.escape(run_dir), "*.png")))
    return [os.path.relpath(f, root) for f in files]


def safe_path(root, rel_path):
    """Absolute path of rel_path inside root (no access outside the results root)."""
    root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(root, rel_path or ""))
    if path != root and not path.startswith(root + os.sep):
        raise UIError("The path is not inside the results directory")
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
# local runs
# ----------------------------------------------------------------------------------------------------------------

class LocalQueue:
    """Runs submitted from the interface, executed one after the other in a background thread. Every run is one
    submission (several environments / tasks = several commands, one after the other), with one log file."""

    def __init__(self, root):
        self.root = root
        self.runs = []                   # submission order
        self.count = 0
        self.cond = threading.Condition()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, kind, items, argv):
        log_dir = os.path.join(self.root, "ui_local")
        os.makedirs(log_dir, exist_ok=True)
        with self.cond:
            self.count += 1
            run_id = f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{self.count}"
        run = {"id": run_id, "kind": kind, "items": list(items), "status": "queued", "exit_codes": [],
               "commands": [python_command(kind, i, argv) for i in items], "current": None,
               "log": os.path.relpath(os.path.join(log_dir, f"{run_id}_{kind}.log"), self.root),
               "submitted": time.time(), "started": None, "finished": None, "process": None}
        with self.cond:
            self.runs.append(run)
            self.cond.notify()
        return run

    def _worker(self):
        while True:
            with self.cond:
                run = next((r for r in self.runs if r["status"] == "queued"), None)
                if run is None:
                    self.cond.wait()
                    continue
                run["status"], run["started"] = "running", time.time()
            with open(os.path.join(self.root, run["log"]), "a") as f:
                for k, cmd in enumerate(run["commands"]):
                    if run["status"] == "cancelled":
                        break
                    run["current"] = run["items"][k]
                    f.write(f"\n$ {' '.join(shlex.quote(t) for t in cmd)}\n")
                    f.flush()
                    # own process group: cancelling stops the Ray workers started by the command too
                    run["process"] = subprocess.Popen(cmd, cwd=self.root, stdout=f, stderr=subprocess.STDOUT,
                                                      env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                                      start_new_session=True)
                    code = run["process"].wait()
                    run["exit_codes"].append(code)
                    f.write(f"[exit {code}]\n")
                    f.flush()
            with self.cond:
                run["finished"], run["current"], run["process"] = time.time(), None, None
                if run["status"] != "cancelled":
                    run["status"] = "finished"

    def cancel(self, run_id):
        with self.cond:
            run = next((r for r in self.runs if r["id"] == run_id), None)
            if run is None:
                raise UIError("No such run")
            if run["status"] not in ("queued", "running"):
                raise UIError("This run has already finished")
            run["status"] = "cancelled"
            process = run["process"]
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                threading.Timer(15, _kill_group, args=(process,)).start()
            except ProcessLookupError:
                pass
        return {"message": f"Cancelled {run_id}"}

    def list(self):
        with self.cond:
            return [{k: r[k] for k in ("id", "kind", "items", "status", "exit_codes", "current", "log", "submitted",
                                       "started", "finished")} for r in reversed(self.runs)]


def _kill_group(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class App:
    def __init__(self, root, token):
        self.root = os.path.abspath(root)
        self.token = token
        self.queue = LocalQueue(self.root)
        self.plot_lock = threading.Lock()
        self.hostname = socket.gethostname()

    def meta(self):
        from ..plotting import ENV_COLORS, XLIM, YLIM
        return {
            "version": __version__, "root": self.root, "hostname": self.hostname, "python": sys.executable,
            "cpus": os.cpu_count(), "user": os.environ.get("USER", ""),
            "envs": [{"id": k, "short": ENV_SHORT_NAMES[k], "color": ENV_COLORS.get(ENV_SHORT_NAMES[k], "gray"),
                      "cls": ENV_IDS[k]} for k in ENV_IDS],
            "tasks": [{"id": t, "envs": [ENV_SHORT_NAMES[e] for e in envs], "rounds": continual_mod.ROUNDS[t]}
                      for t, envs in continual_mod.TASKS.items()],
            "colors": ENV_COLORS, "xlim": XLIM, "ylim": YLIM,
            "schemas": {kind: parser_schema(c["parser"]()) for kind, c in COMMANDS.items()},
            "algorithms": [{"kind": k, "label": label} for k, label in ALGORITHMS.items()],
        }

    def preview(self, body):
        kind, items = body.get("kind"), body.get("items") or []
        argv, _ = build_argv(kind, body.get("values"))
        if not items:
            raise UIError("Select at least one environment / task")
        bad = [i for i in items if i not in COMMANDS[kind]["items"]]
        if bad:
            raise UIError(f"unknown items {bad}")
        commands = [" ".join(shlex.quote(t) for t in python_command(kind, i, argv)) for i in items]
        return {"argv": argv, "commands": commands}

    def submit(self, body):
        p = self.preview(body)
        run = self.queue.submit(body["kind"], body["items"], p["argv"])
        waiting = sum(r["status"] in ("queued", "running") for r in self.queue.list()) - 1
        where = "started" if waiting == 0 else f"queued ({waiting} ahead)"
        return {"message": f"Run {run['id']} {where}: {len(body['items'])} commands, one after the other", "id": run["id"],
                "log": run["log"]}

    def read_log(self, path, lines=200):
        """Last lines of a log file inside the results root."""
        real = safe_path(self.root, path)
        if not os.path.isfile(real):
            raise UIError("The log file does not exist")
        with open(real, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 400_000))
            text = f.read().decode("utf-8", "replace")
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        return {"path": os.path.relpath(real, self.root), "text": "\n".join(text.splitlines()[-int(lines):]),
                "size": size}

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
                return self._send(403, {"error": "Invalid token: use the full address opened / printed by masc-ui at start-up (with ?token=...)"})
            q = {k: v[0] for k, v in query.items()}
            try:
                body = {}
                if method == "POST":
                    length = int(self.headers.get("Content-Length") or 0)
                    try:
                        body = json.loads(self.rfile.read(length) or b"{}")
                    except ValueError:
                        raise UIError("The request is not valid JSON")
                route = url.path[len("/api/"):]
                if route == "file":
                    path = safe_path(app.root, q.get("path"))
                    if not os.path.isfile(path):
                        raise UIError("The file does not exist")
                    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
                    with open(path, "rb") as f:
                        return self._send(200, f.read(), ctype)
                handlers = {
                    ("GET", "meta"): app.meta,
                    ("POST", "preview"): lambda: app.preview(body),
                    ("POST", "submit"): lambda: app.submit(body),
                    ("GET", "runs_local"): lambda: {"runs": app.queue.list()},
                    ("POST", "cancel"): lambda: app.queue.cancel(str(body.get("id", ""))),
                    ("GET", "log"): lambda: app.read_log(q.get("path", ""), int(q.get("lines", 200))),
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


def serve(root=".", port=8765, token=None):
    """Start the server on 127.0.0.1 (first free port from `port`, 0: any free port); returns (server, app, url)."""
    app = App(root, token or secrets.token_urlsafe(16))
    handler = make_handler(app)
    for p in ([0] if port == 0 else range(port, port + 20)):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port in {port}-{port + 19}")
    url = f"http://localhost:{server.server_address[1]}/?token={app.token}"
    return server, app, url


def check_platform():
    """masc-ui is meant for a local Mac only (not for Iridis: use the commands and jobs/ there)."""
    if sys.platform != "darwin" and not os.environ.get("MASC_UI_ANY_PLATFORM"):
        raise SystemExit(
            f"masc-ui runs only locally on a Mac (this computer: {sys.platform}, {socket.gethostname()}).\n"
            "On Iridis, use the commands (masc-train, masc-continual, masc-plot) and the SLURM scripts of jobs/; "
            "copy the results to the Mac (rsync) to browse them in masc-ui.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="masc-ui", description="local web interface of the masc package (Mac)")
    parser.add_argument("--root", default=".", help="results directory: the runs start and write there, its results "
                                                     "are shown (default: current directory)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args(argv)
    check_platform()
    os.makedirs(args.root, exist_ok=True)
    server, app, url = serve(args.root, args.port)
    print(f"masc-ui {__version__}, results root {app.root}")
    print(f"  open: {url}")
    print("  stop with Ctrl-C (runs that are still running are stopped too)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        for run in app.queue.runs:
            if run["process"] is not None and run["process"].poll() is None:
                _kill_group(run["process"])
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
