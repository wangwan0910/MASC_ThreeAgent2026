# -*- coding: utf-8 -*-
"""
Multi-agent (Ray RLlib) wrapper of the serial supply chain environments.

The single-planner environments (AssembySto.Stocha_P*) keep the state as a
(n_rows, n_stage) matrix and the action as a (3, n_stage) matrix, one column per stage.
SupplyChainMARLEnv splits these matrices by column: every stage (factory, warehouse, retailer) is one
RLlib agent (heterogeneous agents, independent PPO), with its own
    observation : its own column of the state matrix
    action      : its own column of the action matrix (own MultiDiscrete space)
    reward      : minus its own cost (inventory holding cost + stockout cost)
    policy      : its own RLModule, no parameter sharing

Event driven: the actions are only executed on reorder days (the retailer's inventory hits its reorder point),
so the agents only act on these days. One RLlib step = one decision:
    obs at a reorder day -> actions executed -> days without decision are simulated -> next reorder day:
    reward = minus the cost of each agent on that day (sparse reward of the original environment)
The days without decision produce no PPO sample, so no log-prob gradient is computed for actions that are not executed.
gamma therefore discounts per decision, not per day.

env_config keys
---------------
env_class    : "module:Class" of the wrapped single-planner environment, default "AssembySto:Stocha_P1"
env_kwargs   : kwargs passed to the environment class
reward_scale : rewards are multiplied by this factor (the costs are in the order of 1e4)
obs_rows     : pad the state matrix with zero rows up to this number of rows, so that environments with
               different state sizes give the same observation space and a trained policy can be transferred
               between them (continual learning)
"""

import importlib
import os
import sys

import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env


ENV_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_NAME = "SupplyChainMARL"
DEFAULT_AGENT_NAMES = ["factory", "warehouse", "retailer"]

# environment ids (as in the original stable-baselines3 scripts)
ENV_IDS = {
    "StoSupplyEnv-v1": "AssembySto:Stocha_P0",     # demand always 2 (std 0)
    "StoSupplyEnv-v2": "AssembySto:Stocha_P01",   # std 0.1
    "StoSupplyEnv-v3": "AssembySto:Stocha_P1",     # std 1
}
# short names used in the figure names and legends, e.g. PPO_Sto1.png, "PPO in Sto1"
ENV_SHORT_NAMES = {"StoSupplyEnv-v1": "Sto0", "StoSupplyEnv-v2": "Sto01", "StoSupplyEnv-v3": "Sto1"}


def resolve_env_class(name):
    """'module:Class' or one of the ids in ENV_IDS -> environment class (module from this directory)."""
    if ENV_DIR not in sys.path:
        sys.path.insert(0, ENV_DIR)
    name = ENV_IDS.get(name, name)
    module_name, class_name = name.split(":")
    return getattr(importlib.import_module(module_name), class_name)


class SupplyChainMARLEnv(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()
        config = dict(config or {})
        self.env = resolve_env_class(config.get("env_class", "AssembySto:Stocha_P1"))(**config.get("env_kwargs", {}))
        self.reward_scale = config.get("reward_scale", 1.0)

        self.n_stage = getattr(self.env, "n_stage", 3)
        obs_nvec = np.asarray(self.env.observation_space.nvec).reshape(-1, self.n_stage)
        act_nvec = np.asarray(self.env.action_space.nvec).reshape(-1, self.n_stage)
        self.n_rows = obs_nvec.shape[0]
        self.obs_rows = config.get("obs_rows") or self.n_rows
        assert self.obs_rows >= self.n_rows, "obs_rows must be >= number of rows of the state matrix"
        self.n_act_rows = act_nvec.shape[0]
        # observations are scaled to [0, 1] with the number of discrete values of each field
        self.obs_scale = np.maximum(obs_nvec - 1, 1).astype(np.float32)

        if hasattr(self.env, "agents"):
            names = [a.name for a in self.env.agents]
        else:
            names = DEFAULT_AGENT_NAMES[:self.n_stage]
        self.possible_agents = list(names)
        self.agents = list(names)

        self.observation_spaces = {
            aid: Box(0.0, 1.0, shape=(self.obs_rows,), dtype=np.float32) for aid in self.possible_agents}
        self.action_spaces = {
            aid: MultiDiscrete(act_nvec[:, k].astype(np.int64)) for k, aid in enumerate(self.possible_agents)}

    def _obs(self, flat_state):
        # each agent observes its own column of the state matrix
        s = np.asarray(flat_state, dtype=np.float32).reshape(self.n_rows, self.n_stage)
        s = np.clip(s / self.obs_scale, 0.0, 1.0)
        if self.obs_rows > self.n_rows:
            s = np.vstack([s, np.zeros((self.obs_rows - self.n_rows, self.n_stage), dtype=np.float32)])
        return {aid: s[:, k].copy() for k, aid in enumerate(self.possible_agents)}

    def _advance(self):
        """Simulate the days without decision until the retailer triggers an order or the episode ends.
        Returns (trigger, terminated, truncated); the state of that day is self.env.state1."""
        no_action = np.zeros(self.n_act_rows * self.n_stage, dtype=np.int64)
        while True:
            trigger = bool(self.env.begin_day())
            terminated = bool(self.env.is_terminated())
            truncated = bool(self.env.is_truncated()) and not terminated
            if trigger or terminated or truncated:
                return trigger, terminated, truncated
            self.env.end_day(no_action)

    def _costs(self, trigger):
        # sparse reward: costs only on reorder days, 0 if the episode ends before the next reorder day
        if not trigger:
            return [0.0] * self.n_stage
        return [float(c) for c in self.env.agent_costs()]

    def reset(self, *, seed=None, options=None):
        for _ in range(100):
            self.env.reset(seed=seed, options=options)
            seed = None
            trigger, terminated, truncated = self._advance()
            if trigger and not (terminated or truncated):
                break
        else:
            raise RuntimeError("no reorder day before the end of the episode")
        self.agents = list(self.possible_agents)
        # cost of the first reorder day (caused by the initial state, not by any action): not a reward,
        # only reported so that evaluate() gives the same episode return as the original environment
        team_reward = -sum(self._costs(True))
        return self._obs(self.env.state1), {aid: {"team_reward": team_reward} for aid in self.possible_agents}

    def step(self, action_dict):
        # assemble the (3, n_stage) action matrix of the original environment from the agents' actions
        action = np.zeros((self.n_act_rows, self.n_stage), dtype=np.int64)
        for k, aid in enumerate(self.possible_agents):
            if aid in action_dict:
                action[:, k] = np.asarray(action_dict[aid]).reshape(-1)

        # reorder day: execute the actions, then go to the next reorder day
        self.env.end_day(action.flatten())
        trigger, terminated, truncated = self._advance()

        costs = self._costs(trigger)
        rewards = {aid: -c * self.reward_scale for aid, c in zip(self.possible_agents, costs)}
        terminateds = {aid: terminated for aid in self.possible_agents}
        truncateds = {aid: truncated for aid in self.possible_agents}
        terminateds["__all__"], truncateds["__all__"] = terminated, truncated
        infos = {aid: {"team_reward": -sum(costs)} for aid in self.possible_agents}
        return self._obs(self.env.state1), rewards, terminateds, truncateds, infos


def env_creator(env_config):
    return SupplyChainMARLEnv(env_config)


register_env(ENV_NAME, env_creator)


def init_ray(**kwargs):
    """ray.init with this directory (marl_env, environments) on the PYTHONPATH of all Ray workers.
    On SLURM, Ray uses the CPUs allocated to the job (--cpus-per-task), not all the CPUs of the node, and every job
    has its own Ray instance."""
    import ray
    if "num_cpus" not in kwargs and os.environ.get("SLURM_CPUS_PER_TASK"):
        kwargs["num_cpus"] = int(os.environ["SLURM_CPUS_PER_TASK"])
    if os.environ.get("SLURM_JOB_ID"):
        # several jobs (e.g. the tasks of a job array) can run on the same node: one Ray instance per job
        job = os.environ["SLURM_JOB_ID"] + "_" + os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        kwargs.setdefault("_temp_dir", f"/tmp/ray_{job}")
        kwargs.setdefault("include_dashboard", False)
    pythonpath = os.pathsep.join([ENV_DIR, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    ray.init(runtime_env={"env_vars": {"PYTHONPATH": pythonpath}}, ignore_reinit_error=True, **kwargs)


def build_ppo_config(env_config, lr=3e-4, hiddens=(64, 64), use_lstm=False, lstm_cell_size=64, max_seq_len=20,
                     num_env_runners=1, num_envs_per_env_runner=1, num_gpus=0,
                     train_batch_size=2048, minibatch_size=64, num_epochs=10):
    """Multi-agent PPO config: one independent policy (RLModule) per supply chain stage.
    The PPO hyperparameters default to the stable-baselines3 PPO defaults used before."""
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    agent_ids = SupplyChainMARLEnv(env_config).possible_agents
    model_config = DefaultModelConfig(
        fcnet_hiddens=list(hiddens), fcnet_activation="tanh", vf_share_layers=False,
        use_lstm=use_lstm, lstm_cell_size=lstm_cell_size, max_seq_len=max_seq_len)

    return (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_config)
        .env_runners(num_env_runners=num_env_runners, num_envs_per_env_runner=num_envs_per_env_runner)
        .learners(num_learners=0, num_gpus_per_learner=num_gpus)
        .training(lr=lr, gamma=0.99, lambda_=0.95, clip_param=0.2, entropy_coeff=0.0, vf_loss_coeff=0.5,
                  grad_clip=0.5, vf_clip_param=float("inf"),
                  train_batch_size_per_learner=train_batch_size, minibatch_size=minibatch_size,
                  num_epochs=num_epochs)
        .multi_agent(policies=set(agent_ids),
                     policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id)
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs={
            aid: RLModuleSpec(model_config=model_config) for aid in agent_ids}))
    )


def evaluate(algo, env_config, n_steps=1000):
    """Run the trained (deterministic) policies for n_steps decisions, return the mean episode team reward
    (total cost of the supply chain, same as the episode return of the original single-planner environment).
    Only for feed-forward policies (no LSTM state)."""
    import torch
    env = SupplyChainMARLEnv(env_config)
    modules = {aid: algo.get_module(aid) for aid in env.possible_agents}
    obs, infos = env.reset()
    returns, ep_return = [], infos[env.possible_agents[0]]["team_reward"]
    for _ in range(n_steps):
        actions = {}
        for aid, module in modules.items():
            out = module.forward_inference({"obs": torch.from_numpy(obs[aid][None])})
            dist = module.get_inference_action_dist_cls().from_logits(out["action_dist_inputs"])
            actions[aid] = dist.to_deterministic().sample()[0].numpy()
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        ep_return += infos[env.possible_agents[0]]["team_reward"]
        if terminateds["__all__"] or truncateds["__all__"]:
            returns.append(ep_return)
            obs, infos = env.reset()
            ep_return = infos[env.possible_agents[0]]["team_reward"]
    return float(np.mean(returns)) if returns else ep_return


if __name__ == '__main__':
    # random actions, just to check that the multi-agent environment runs through episodes
    for env_class in ENV_IDS.values():
        env = SupplyChainMARLEnv({"env_class": env_class})
        obs, _ = env.reset(seed=0)
        done, decisions = False, 0
        while not done:
            decisions += 1
            actions = {aid: env.action_spaces[aid].sample() for aid in env.agents}
            obs, rew, term, trunc, info = env.step(actions)
            assert all(env.observation_spaces[aid].contains(o) for aid, o in obs.items())
            done = term["__all__"] or trunc["__all__"]
        print(env_class, "ok, days:", env.env.day, "decisions:", decisions)
