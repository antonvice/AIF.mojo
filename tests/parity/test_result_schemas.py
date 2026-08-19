"""Golden tests for the four experiment JSON contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiment_matrix import (  # noqa: E402
    ENVIRONMENTS,
    ENVIRONMENT_ACTIONS,
    METHODS,
    make_result_document,
    parse_native_policies,
    validate_result_document,
    write_matrix,
)


GOLDEN_KEYS = {
    "frozen_lake": {
        "config": (
            "environment",
            "grid_size",
            "n_configs",
            "hole_fraction",
            "base_noise",
            "noise_range",
            "slip_prob",
            "hole_penalty",
            "goal_temperature",
            "scan_cost",
            "planning_method",
            "n_episodes",
            "max_steps",
            "planning_horizon",
            "planning_iterations",
            "receding_horizon",
            "seed_start",
        ),
        "results": (
            "success_rate",
            "avg_steps",
            "avg_reward",
            "successes",
            "total_time_s",
        ),
    },
    "wumpus_world": {
        "config": (
            "environment",
            "grid_size",
            "n_configs",
            "n_pits",
            "obs_noise",
            "pos_noise",
            "slip_prob",
            "sense_cost",
            "planning_method",
            "n_episodes",
            "max_steps",
            "planning_horizon",
            "planning_iterations",
            "receding_horizon",
            "seed_start",
        ),
        "results": (
            "success_rate",
            "avg_steps",
            "avg_reward",
            "successes",
            "total_time_s",
        ),
    },
    "rocksample": {
        "config": (
            "environment",
            "grid_size",
            "n_rocks",
            "n_configs",
            "half_eff_dist",
            "pos_noise",
            "slip_prob",
            "good_reward",
            "bad_penalty",
            "exit_reward",
            "good_logit",
            "bad_logit",
            "exit_logit",
            "goal_temperature",
            "sense_cost",
            "sample_cost",
            "planning_method",
            "n_episodes",
            "max_steps",
            "planning_horizon",
            "planning_iterations",
            "receding_horizon",
            "seed_start",
        ),
        "results": (
            "success_rate",
            "avg_steps",
            "avg_reward",
            "std_reward",
            "good_rock_retrieval",
            "total_good_rocks",
            "total_good_collected",
            "episodes_with_good_rocks",
            "successes",
            "total_time_s",
        ),
    },
    "minigrid": {
        "config": (
            "grid_size",
            "env_name",
            "planning_method",
            "fov_size",
            "n_episodes",
            "max_steps",
            "planning_horizon",
            "receding_horizon",
            "inference_iterations",
            "planning_iterations",
            "no_orientation",
            "damping",
            "obs_alpha",
            "seed_start",
        ),
        "results": (
            "success_rate",
            "avg_steps",
            "avg_reward",
            "successes",
            "total_time_s",
        ),
    },
}


@pytest.mark.parametrize(
    ("environment", "method"),
    (
        ("frozen_lake", "loopy"),
        ("wumpus_world", "dyn-channel"),
        ("rocksample", "active-inference"),
        ("minigrid", "region-extended"),
    ),
)
def test_golden_environment_schema(environment: str, method: str) -> None:
    document = make_result_document(environment, method, elapsed=0.125)
    round_tripped = json.loads(json.dumps(document))
    validate_result_document(environment, round_tripped)
    assert tuple(round_tripped) == ("config", "results")
    assert tuple(round_tripped["config"]) == GOLDEN_KEYS[environment]["config"]
    assert tuple(round_tripped["results"]) == GOLDEN_KEYS[environment]["results"]
    assert round_tripped["config"]["planning_method"] == method


def test_retained_method_names_are_exact() -> None:
    assert METHODS == (
        "loopy-vbp",
        "loopy",
        "region-extended",
        "dyn-channel",
        "nuijten",
        "vbp-channel",
        "precise-info-seeking",
        "active-inference",
    )


def test_schema_rejects_extra_fields_and_bool_as_integer() -> None:
    document = make_result_document("frozen_lake", "loopy")
    document["results"]["native_policy"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="fields changed"):
        validate_result_document("frozen_lake", document)
    document = make_result_document("frozen_lake", "loopy")
    document["config"]["grid_size"] = True
    with pytest.raises(TypeError, match="grid_size must be int"):
        validate_result_document("frozen_lake", document)


def test_native_policy_protocol_covers_all_methods() -> None:
    output = "\n".join(
        f"ACTION {method} {action} {probability}"
        for method in METHODS
        for action, probability in enumerate((0.2, 0.3, 0.5))
    )
    policies = parse_native_policies(output)
    assert tuple(policies) == METHODS
    assert all(policy == [0.2, 0.3, 0.5] for policy in policies.values())


def test_bounded_matrix_writes_32_valid_results(tmp_path: Path) -> None:
    runs = {
        environment: (
            {
                method: [1.0 / ENVIRONMENT_ACTIONS[environment]]
                * ENVIRONMENT_ACTIONS[environment]
                for method in METHODS
            },
            0.08,
        )
        for environment in ENVIRONMENTS
    }
    summary_path = write_matrix(tmp_path, runs, compile_seconds=1.25)
    summary = json.loads(summary_path.read_text())
    assert summary["case_count"] == 32
    assert len(summary["cases"]) == 32
    result_paths = sorted(tmp_path.glob("*/*.json"))
    assert len(result_paths) == 32
    for path in result_paths:
        validate_result_document(path.parent.name, json.loads(path.read_text()))
