"""Web interface (masc.ui.server): forms, commands, results, simulation, HTTP token."""
import json
import os
import threading
import urllib.error
import urllib.request

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


def test_sbatch_script():
    argv, _ = ui.build_argv("continual", {"--rounds": "1"})
    script = ui.sbatch_script("continual", ["0-01-1", "sto1-bat10"], argv, "/tmp/runs", ui.default_slurm("continual"))
    assert "#SBATCH --array=0-1" in script and "ITEMS=(0-01-1 sto1-bat10)" in script
    assert '-m masc.continual --task "$ITEM" --rounds 1' in script
    with pytest.raises(ui.UIError):
        ui.sbatch_script("train", ["BatSupplyEnv-v1"], [], "/tmp", {**ui.default_slurm("train"), "mem": "1G; rm -rf"})


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
    server, app, url = ui.serve(tmp_path, port=0, open_message=False)
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
