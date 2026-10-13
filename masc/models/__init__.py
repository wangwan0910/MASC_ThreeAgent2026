# -*- coding: utf-8 -*-
"""
Pluggable policy / value networks of the agents (one RLModule per supply chain stage, see marl_env.build_ppo_config).

Every architecture is registered by name with its default settings; build_ppo_config(model=<name>,
model_config={...}) builds the RLModuleSpec of every agent from the registry:

    mlp          feed-forward network (PPO, masc-train)                         RLlib default module
    transformer  gated transformer over the last observations (masc-train-transformer), masc/models/transformer.py

Adding an architecture: write a TorchRLModule (+ ValueFunctionAPI for PPO), register a builder that returns its
RLModuleSpec, e.g.

    from masc.models import register_model

    @register_model("gru", defaults={"hidden": 64, "max_seq_len": 20})
    def gru_spec(config):
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        return RLModuleSpec(module_class=MyGRUModule, model_config=config)

and use it with build_ppo_config(env_config, model="gru", model_config={"hidden": 128}) or a small entry script as
masc/main_transformerPPO.py.
"""

MODELS = {}     # name -> {"builder": function(config dict) -> RLModuleSpec, "defaults": dict, "doc": str}


def register_model(name, defaults=None):
    """Decorator: register a builder function(config) -> RLModuleSpec under `name`, with its default settings
    (the settings a user can change; others are rejected by build_rl_module_spec)."""
    def decorator(builder):
        if name in MODELS:
            raise ValueError(f"model {name!r} is already registered")
        MODELS[name] = {"builder": builder, "defaults": dict(defaults or {}), "doc": (builder.__doc__ or "").strip()}
        return builder
    return decorator


def model_names():
    return sorted(MODELS)


def model_defaults(name):
    return dict(_get(name)["defaults"])


def build_rl_module_spec(name, model_config=None):
    """RLModuleSpec of the registered model `name`, with its defaults updated by model_config (None values are
    ignored, i.e. keep the default)."""
    entry = _get(name)
    overrides = {k: v for k, v in (model_config or {}).items() if v is not None}
    unknown = set(overrides) - set(entry["defaults"])
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)} for model {name!r}, expected {sorted(entry['defaults'])}")
    return entry["builder"]({**entry["defaults"], **overrides})


def _get(name):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, registered: {model_names()}")
    return MODELS[name]


# ----- RLlib default modules -----

@register_model("mlp", defaults={"hiddens": (64, 64), "activation": "tanh"})
def mlp_spec(config):
    """Feed-forward policy and value networks (separate), RLlib default PPO module."""
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    return RLModuleSpec(model_config=DefaultModelConfig(
        fcnet_hiddens=list(config["hiddens"]), fcnet_activation=config["activation"], vf_share_layers=False))


# architectures defined in their own modules register themselves on import
from . import transformer  # noqa: E402,F401
