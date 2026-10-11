"""Failure-aware summaries for repeated samples of one fixed DGP.

Errors are summarized only on explicitly successful, finite-error records.
Paired differences use the same trials for both methods. These Monte Carlo
standard errors describe repeated samples conditional on a DGP, not variation
across independently generated DGPs or uncertainty of a causal coefficient.
"""

import csv
import json
from numbers import Real
from pathlib import Path

import numpy as np


_VALID_STATUSES = {"ok", "underidentified", "failed"}
_ERROR_SCOPE = "conditional_on_status_ok_and_finite_nonnegative_l2_error"
_PAIRED_SCOPE = "conditional_on_both_methods_usable_in_the_same_trial"


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _false(value):
    return isinstance(value, (bool, np.bool_)) and not bool(value)


def _record_info(record):
    if record is None:
        return {"usable": False, "error": None, "failed": True, "missing": True,
                "invalid_status": False, "invalid_error": False}
    if not isinstance(record, dict):
        raise ValueError("Each method record must be a dictionary")
    status = record.get("status")
    error = _number(record.get("l2_error"))
    invalid_error = status == "ok" and (error is None or error < 0)
    invalid_status = status not in _VALID_STATUSES
    return {
        "usable": status == "ok" and not invalid_error,
        "error": error, "missing": False,
        "failed": status == "failed" or invalid_status or invalid_error,
        "invalid_status": invalid_status, "invalid_error": invalid_error,
    }


def _mean_mcse(values):
    """Scale first to avoid overflow when summing finite observations."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    scale = float(np.max(np.abs(values)))
    normalized = values / scale if scale else values
    mean = float(scale * np.mean(normalized)) if scale else 0.0
    if len(values) < 2:
        return mean, None
    mcse = float(scale * (np.std(normalized, ddof=1) / np.sqrt(len(values)))) if scale else 0.0
    return mean, mcse


def _error_summary(errors):
    errors = np.asarray(errors, dtype=float)
    if not len(errors):
        return {name: None for name in ("mean_l2", "mean_l2_mcse", "rmse", "median", "p95", "p99", "max")}
    maximum = float(errors.max())
    scaled = errors / maximum if maximum else errors
    mean, mcse = _mean_mcse(errors)
    return {
        "mean_l2": mean, "mean_l2_mcse": mcse,
        "rmse": float(maximum * np.sqrt(np.mean(scaled**2))) if maximum else 0.0,
        "median": float(maximum * np.quantile(scaled, .5)),
        "p95": float(maximum * np.quantile(scaled, .95)),
        "p99": float(maximum * np.quantile(scaled, .99)), "max": maximum,
    }


def _recorded_mean(records, field):
    values = [_number(record.get(field)) for record in records if record is not None]
    values = [value for value in values if value is not None]
    mean, _ = _mean_mcse(values)
    return mean, len(values)


def _rescale_squared(value, scale):
    if value is None:
        return None
    # Multiplying in this order retains a representable small difference even
    # when scale**2 itself overflows. Unrepresentable statistics remain explicit.
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.float64(scale) * (np.float64(scale) * value))
    return result if np.isfinite(result) else None


def summarize_trials(trials, primary="no_pilot_random_equal_budget"):
    """Summarize one scenario/DGP with explicit exclusion and pairing counts.

    Missing records and malformed successful errors count as failures. A status
    of underidentified is excluded but is not a numerical failure. Population-
    and sample-rank counters are separate and may overlap. Costs and sample
    lengths include every finite recorded attempt, including failed attempts.
    MCSE is None for fewer than two usable observations, never silently zero.
    """
    trials = list(trials)
    ids = []
    methods = set()
    for trial in trials:
        if not isinstance(trial, dict) or "trial_id" not in trial or not isinstance(trial.get("methods"), dict):
            raise ValueError("Each trial needs trial_id and a methods dictionary")
        ids.append(trial["trial_id"])
        methods.update(trial["methods"])
    try:
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate trial IDs would count dependent repetitions as independent trials")
    except TypeError as error:
        raise ValueError("trial_id must be hashable") from error
    if not trials:
        return {"methods": {}, "paired_comparisons": {}}
    methods.add(primary)
    if any(not isinstance(method, str) for method in methods):
        raise ValueError("Method names must be strings")
    count = len(trials)
    records = {method: [trial["methods"].get(method) for trial in trials] for method in sorted(methods)}
    inspected = {method: [_record_info(record) for record in series] for method, series in records.items()}
    summaries = {}
    for method, series in records.items():
        info = inspected[method]
        errors = [item["error"] for item in info if item["usable"]]
        mean_cost, cost_count = _recorded_mean(series, "total_cost")
        mean_n, n_count = _recorded_mean(series, "n_main")
        summaries[method] = {
            "attempted": count, "valid_count": len(errors), "excluded_count": count - len(errors),
            "population_underidentified_count": sum(_false(record.get("population_full_rank")) for record in series if record is not None),
            "sample_under_rank_count": sum(_false(record.get("sample_full_rank")) for record in series if record is not None),
            "underidentified_status_count": sum(record.get("status") == "underidentified" for record in series if record is not None),
            "failure_count": sum(item["failed"] for item in info),
            "missing_record_count": sum(item["missing"] for item in info),
            "invalid_status_count": sum(item["invalid_status"] for item in info),
            "invalid_ok_error_count": sum(item["invalid_error"] for item in info),
            **_error_summary(errors),
            "mean_cost": mean_cost, "cost_recorded_count": cost_count, "cost_missing_count": count - cost_count,
            "mean_n_main": mean_n, "n_main_recorded_count": n_count, "n_main_missing_count": count - n_count,
            "error_summary_scope": _ERROR_SCOPE,
            "cost_summary_scope": "all_attempts_with_finite_recorded_values",
        }
    paired = {}
    primary_info = inspected[primary]
    for method, info in inspected.items():
        if method == primary:
            continue
        shared = [(left["error"], right["error"]) for left, right in zip(info, primary_info)
                  if left["usable"] and right["usable"]]
        if shared:
            values = np.asarray(shared, dtype=float)
            l2_mean, l2_mcse = _mean_mcse(values[:, 0] - values[:, 1])
            scale = float(values.max())
            scaled = values / scale if scale else values
            squared_mean, squared_mcse = _mean_mcse((scaled[:, 0] - scaled[:, 1]) * (scaled[:, 0] + scaled[:, 1]))
            squared_mean = _rescale_squared(squared_mean, scale)
            squared_mcse = _rescale_squared(squared_mcse, scale)
        else:
            l2_mean = l2_mcse = squared_mean = squared_mcse = None
        paired[method] = {
            "primary": primary, "attempted": count, "paired_count": len(shared), "excluded_count": count - len(shared),
            "method_valid_count": summaries[method]["valid_count"], "primary_valid_count": summaries[primary]["valid_count"],
            "method_only_valid_count": sum(left["usable"] and not right["usable"] for left, right in zip(info, primary_info)),
            "primary_only_valid_count": sum(right["usable"] and not left["usable"] for left, right in zip(info, primary_info)),
            "neither_valid_count": sum(not left["usable"] and not right["usable"] for left, right in zip(info, primary_info)),
            "mean_squared_error_difference": squared_mean, "squared_error_difference_mcse": squared_mcse,
            "mean_l2_difference": l2_mean, "l2_difference_mcse": l2_mcse,
            "squared_mean_unrepresentable": bool(shared) and squared_mean is None,
            "squared_mcse_unrepresentable": len(shared) >= 2 and squared_mcse is None,
            "comparison_scope": _PAIRED_SCOPE,
            "difference_direction": "method_minus_primary; negative favors method",
            "mcse_interpretation": "Monte Carlo standard error conditional on this fixed DGP; undefined for fewer than two pairs",
        }
    return {"methods": summaries, "paired_comparisons": paired}


def write_benchmark_tables(result, output_dir):
    """Write one row per method/comparison and DGP; never pool DGP replicates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_rows, paired_rows = [], []
    for entry in result["scenarios"]:
        scenario = entry["scenario"]
        if isinstance(scenario, (dict, list)):
            scenario = json.dumps(scenario, sort_keys=True)
        prefix = {"scenario": scenario, "dgp_seed": entry["dgp_seed"]}
        for method, summary in entry["summary"]["methods"].items():
            method_rows.append({**prefix, "method": method, **summary})
        for method, summary in entry["summary"]["paired_comparisons"].items():
            paired_rows.append({**prefix, "method": method, **summary})
    paths = {}
    for name, rows in (("method_summary", method_rows), ("paired_comparisons", paired_rows)):
        path = output_dir / (name + ".csv")
        fields = ["scenario", "dgp_seed", "method"]
        for row in rows:
            fields.extend(key for key in row if key not in fields)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        paths[name + "_csv"] = str(path)
    return paths
