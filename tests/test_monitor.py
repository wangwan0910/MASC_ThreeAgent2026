"""TrainingMonitor stopping rules on synthetic reward curves."""
import numpy as np

from masc.train import TrainingMonitor


def feed(monitor, curve, steps_per_iter=2048):
    for i, ret in enumerate(curve, start=1):
        if monitor("t", {"num_env_steps_sampled_lifetime": i * steps_per_iter,
                         "env_runners": {"episode_return_mean": ret}}):
            return i
    return None


def learning_curve(n=1500, seed=0):
    x = np.arange(n)
    return -100 + 80 * (1 - np.exp(-x / 100)) + np.random.RandomState(seed).normal(0, 0.3, n)


def test_converges_on_a_plateau():
    m = TrainingMonitor(50, 0.01, 100, 100000, 1e9, 1e9)
    assert feed(m, learning_curve()) is not None
    assert m.status["t"]["status"] == "converged"


def test_no_early_stop_records_the_verdict():
    m = TrainingMonitor(50, 0.01, 100, 100000, 1500 * 2048, 1e9, early_stop=False)
    assert feed(m, learning_curve()) == 1500                     # runs to max_timesteps
    assert m.status["t"]["status"] == "completed"
    assert "converged" in m.status["t"]["would_stop"]


def test_collapse():
    curve = np.concatenate([learning_curve(600), np.full(600, -90.0)])
    m = TrainingMonitor(50, 0.01, 10000, 100000, 1e9, 1e9)
    feed(m, curve)
    assert m.status["t"]["status"] == "collapsed"
