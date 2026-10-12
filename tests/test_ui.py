"""Web interface (masc.ui.server): forms, commands, local queue, results, simulation, HTTP token, platform."""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from masc.ui import server as ui


def test_schema_hides_item_and_out_dir():
    options = {f["option"] for f in ui.parser_schema(ui.COMMANDS["train"]["parser"]())}
    assert {"--lr", "--seeds", "--no-early-stop", "--no-stockout-termination"} <= options
    assert not {"--env", "--out-dir"} & options


def test_build_argv_only_changed_values():
    argv, ns = ui.build_argv("train", {"--lr": "0.0001", "--seeds": "0, 1 2", "--no-early-stop": True,
                                       "--reward-mode": "team", "--vf-clip": "10.0", "--tag": ""})
    assert argv == ["--lr", "0.0001", "--seeds", "0", "1", "2", "--no-early-stop"]
    assert ns.seeds == [0, 1, 2] and ns.early_stop is False


def test_build_argv_uses_the_defaults_of_the_command():
    assert ui.build_argv("train_rppo", {"--dir": "tu1"})[0] == []
    assert ui.build_argv("train", {"--dir": "tu1"})[0] == ["--dir", "tu1"]


@pytest.mark.parametrize("values", [{"--lr": "abc"}, {"--reward-mode": "nobody"}, {"--unknown": "1"}])
def test_build_argv_rejects_bad_values(values):
    with pytest.raises(ui.UIError):
        ui.build_argv("train", values)


def test_safe_path(tmp_path):
    assert ui.safe_path(tmp_path, "a/b") == os.path.join(os.path.realpath(tmp_path), "a", "b")
    with pytest.raises(ui.UIError):
        ui.safe_path(tmp_path, "../outside")


def make_run(root, seeds=(0, 1)):
    run = root / "tu_seeds_logs" / "Bat10" / "1791800000"
    for seed in seeds:
        trial = run / "PPO_5seeds" / f"PPO_SupplyChainMARL_x_0000{seed}_seed={seed}"
        trial.mkdir(parents=True)
        (trial / "params.json").write_text(json.dumps({"lr": 0.0001, "seed": seed}))
        rows = ["num_env_steps_sampled_lifetime,env_runners/team_return"]
        rows += [f"{2048 * i},{-1e6 / i - seed * 1000}" for i in range(1, 51)]
        (trial / "progress.csv").write_text("\n".join(rows))
    (run / "training_status.json").write_text(json.dumps([{"lr": 0.0001, "seed": s, "status": "completed"}
                                                          for s in seeds]))
    return run


def test_runs_and_curves(tmp_path):
    make_run(tmp_path)
    runs = ui.find_runs(tmp_path)
    assert len(runs) == 1 and runs[0]["env"] == "Bat10" and runs[0]["n_trials"] == 2
    assert runs[0]["state"] == "completed"
    curves = ui.run_curves(tmp_path, runs[0]["path"])
    assert len(curves["series"]) == 2 and len(curves["bands"]) == 1
    band = curves["bands"][0]
    assert band["y"][0] == pytest.approx(-1e6 - 500) and band["hi"][0] - band["y"][0] == pytest.approx(500)


def test_simulate():
    r = ui.simulate("BatSupplyEnv-v1", {"factory": 18, "warehouse": 18, "retailer": 18}, 6, n_seeds=1)
    assert r["n"] == 1 and r["mean"] < 0 and len(r["trace"]["day"]) == 101
    # the daily costs of the trace add up to the episode total cost
    assert sum(r["trace"]["cost"]) == pytest.approx(-r["mean"])
    with pytest.raises(ui.UIError):
        ui.simulate("nope", {"factory": 1, "warehouse": 1, "retailer": 1}, 1)


def test_http_token(tmp_path):
    server, app, url = ui.serve(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"MASC" in urllib.request.urlopen(base + "/").read()
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/api/meta")
        assert e.value.code == 403
        meta = json.loads(urllib.request.urlopen(f"{base}/api/meta?token={app.token}").read())
        assert len(meta["envs"]) == 6 and "train" in meta["schemas"]
        req = urllib.request.Request(base + "/api/preview", method="POST", headers={"X-Token": app.token},
                                     data=json.dumps({"kind": "train", "items": ["BatSupplyEnv-v1"],
                                                      "values": {"--lr": "0.0001"}}).encode())
        preview = json.loads(urllib.request.urlopen(req).read())
        assert preview["argv"] == ["--lr", "0.0001"] and "masc.train --env BatSupplyEnv-v1" in preview["commands"][0]
    finally:
        server.shutdown()
        server.server_close()


def wait_until(predicate, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def test_local_queue_runs_one_after_the_other(tmp_path, monkeypatch):
    def fake_command(kind, item, argv):
        return [sys.executable, "-c", f"import time; time.sleep(0.5); print('ran {item}'); raise SystemExit({len(item) % 2})"]
    monkeypatch.setattr(ui, "python_command", fake_command)
    queue = ui.LocalQueue(str(tmp_path))
    first = queue.submit("train", ["BatSupplyEnv-v1", "BatSupplyEnv-v2"], [])
    second = queue.submit("continual", ["0-01-1"], [])
    third = queue.submit("continual", ["1-01-0"], [])
    assert queue.cancel(third["id"])["message"]                       # cancelled while queued: never runs
    assert wait_until(lambda: second["status"] == "已完成")
    assert first["finished"] <= second["started"]                      # one after the other
    assert first["exit_codes"] == [1, 1] and second["exit_codes"] == [0]
    log = (tmp_path / first["log"]).read_text()
    assert "ran BatSupplyEnv-v1" in log and "ran BatSupplyEnv-v2" in log and "[exit 1]" in log
    assert third["status"] == "已取消" and third["started"] is None


def test_cancel_running_command(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "python_command", lambda kind, item, argv: [sys.executable, "-c", "import time; time.sleep(60)"])
    queue = ui.LocalQueue(str(tmp_path))
    run = queue.submit("train", ["BatSupplyEnv-v1"], [])
    assert wait_until(lambda: run["process"] is not None)
    queue.cancel(run["id"])
    assert wait_until(lambda: run["finished"] is not None, timeout=20)
    assert run["status"] == "已取消" and run["exit_codes"][0] != 0


def test_mac_only(monkeypatch):
    monkeypatch.delenv("MASC_UI_ANY_PLATFORM", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(SystemExit):
        ui.check_platform()
    monkeypatch.setattr(sys, "platform", "darwin")
    ui.check_platform()


def test_transformer_algorithm():
    schema = {f["option"]: f for f in ui.parser_schema(ui.COMMANDS["train_transformer"]["parser"]())}
    assert schema["--memory"]["group"] == "Transformer" and schema["--memory"]["default"] == 20
    argv, ns = ui.build_argv("train_transformer", {"--memory": "30", "--head-hiddens": "32 32", "--dir": "tu2"})
    assert argv == ["--memory", "30", "--head-hiddens", "32", "32"] and ns.tf_memory == 30
    cmd = ui.python_command("train_transformer", "BatSupplyEnv-v1", argv)
    assert cmd[1:5] == ["-m", "masc.main_transformerPPO", "--env", "BatSupplyEnv-v1"]
    assert list(ui.ALGORITHMS) == ["train", "train_rppo", "train_transformer"]


def test_seed_curves_with_drifting_steps():
    from masc.plotting import align_curves
    a = (np.arange(1, 101) * 2048, np.linspace(-10, 0, 100))
    steps_b = np.arange(1, 101) * 2048
    steps_b[50:] += 1024                                 # one iteration sampled 1024 more steps
    b = (steps_b, np.linspace(-10, 0, 100) - 1)
    grid, values = align_curves([a, b])
    assert grid[-1] == 100 * 2048 and values.shape == (2, 100)
    assert np.allclose(values[1] - values[0], -1, atol=0.06)
