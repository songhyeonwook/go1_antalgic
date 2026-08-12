from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

@dataclass
class ExplorationConfig:
    noise_std_type: str
    init_noise_std: float
    min_action_std: float
    enforce_min_std: bool
    
@dataclass
class EvaluationConfig:
    peg_leg: str = "normal"

@dataclass
class TrainConfig:
    project_name: str
    task: str
    agent: str
    num_envs: int
    max_iterations: int | None
    seed: int

@dataclass
class EnvironmentConfig:
    path: Path
    values: dict[str, Any]
    
@dataclass
class CheckpointConfig:
    mode: str
    teacher: str | None
    student: str | None
    load_optimizer: bool
    reset_iteration: bool

@dataclass
class ExperimentConfig:
    phase: str
    train: TrainConfig
    environment: EnvironmentConfig
    checkpoint: CheckpointConfig
    common: dict[str, Any]
    evaluation: EvaluationConfig
    exploration: ExplorationConfig
    rsl_logger: str = "tensorboard"
    
def deep_merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value

    return base

def read_yaml(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)

    try:
        with config_path.open("r", encoding="utf-8") as file:
            configs = yaml.safe_load(file)

    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Config file does not exist: {config_path}"
        ) from error

    except yaml.YAMLError as error:
        raise ValueError(
            f"Failed to parse YAML file: {config_path}"
        ) from error

    if configs is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(configs, dict):
        raise TypeError(
            f"Config root must be a dictionary: {config_path}"
        )

    return configs

def parse_exploration_config(phase_cfg: dict) -> ExplorationConfig:
    exploration_raw = phase_cfg.get("exploration", {})
    noise_std_raw = exploration_raw.get("noise_std", {})

    # set: scalar 또는 set: log
    selected_type = str(noise_std_raw.get("set", "scalar")).strip().lower()
    valid_types = {"scalar", "log"}

    if selected_type not in valid_types:
        raise ValueError(
            "exploration.noise_std.set must be one of "
            f"{sorted(valid_types)}, got {selected_type!r}"
        )

    profiles = noise_std_raw.get("types", {})

    if not isinstance(profiles, dict):
        raise TypeError(
            "exploration.noise_std.types must be a dictionary"
        )

    if selected_type not in profiles:
        raise KeyError(
            f"Selected noise std type {selected_type!r} does not "
            "exist under exploration.noise_std.types"
        )

    selected_profile = profiles[selected_type]

    if not isinstance(selected_profile, dict):
        raise TypeError(
            f"exploration.noise_std.types.{selected_type} "
            "must be a dictionary"
        )

    init_noise_std = float(selected_profile["init_noise_std"])
    min_action_std = float(selected_profile["min_action_std"])
    enforce_min_std = bool(selected_profile.get("enforce_min_std", True))

    if init_noise_std <= 0.0:
        raise ValueError(
            "init_noise_std must be greater than zero, "
            f"got {init_noise_std}"
        )

    if min_action_std <= 0.0:
        raise ValueError(
            "min_action_std must be greater than zero, "
            f"got {min_action_std}"
        )

    if min_action_std > init_noise_std:
        raise ValueError(
            "min_action_std must not be greater than "
            f"init_noise_std: {min_action_std} > {init_noise_std}"
        )

    return ExplorationConfig(
        noise_std_type=selected_type,
        init_noise_std=init_noise_std,
        min_action_std=min_action_std,
        enforce_min_std=enforce_min_std,
    )

def load_experiment_config(phase_path: str, common_path: str) -> ExperimentConfig:
    phase_path = Path(phase_path).resolve()
    common_path = Path(common_path).resolve()

    phase_cfg = read_yaml(phase_path)
    common_cfg = read_yaml(common_path)

    # train
    train_cfg = TrainConfig(**phase_cfg["train"])
    checkpoint_cfg = CheckpointConfig(**phase_cfg["checkpoint"])
    
    # evaluation
    evaluation_raw = phase_cfg.get("eval", {})
    evaluation_cfg = EvaluationConfig(**evaluation_raw)
    evaluation_cfg.peg_leg = evaluation_cfg.peg_leg.strip().lower()
        
    valid_eval_legs = {
        "normal",
        "fl",
        "fr",
        "rl",
        "rr",
        "balanced",
    }

    if evaluation_cfg.peg_leg not in valid_eval_legs:
        raise ValueError(
            "eval.peg_leg must be one of "
            f"{sorted(valid_eval_legs)}, got {evaluation_cfg.peg_leg!r}"
        )    
            
    # env
    env_section = phase_cfg["env"]
    env_reference = env_section["path"]
    env_path = (phase_path.parent / env_reference).resolve()
    
    environment_values = read_yaml(env_path)
    environment_values = deep_merge(
        environment_values,
        env_section.get("overrides", {})
    )
    
    environment_cfg = EnvironmentConfig(path=env_path, values=environment_values)

    exploration_cfg = parse_exploration_config(phase_cfg)
        
    return ExperimentConfig(
        phase=phase_path.stem,
        train=train_cfg,
        environment=environment_cfg,
        checkpoint=checkpoint_cfg,
        common=common_cfg,
        evaluation=evaluation_cfg,
        exploration=exploration_cfg,
        rsl_logger=common_cfg.get("rsl_logger", "tensorboard"),
    )
