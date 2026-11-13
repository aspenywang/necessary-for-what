"""Fit weighted survey references and apply Own fits to Shuffled inputs.

Input is the full-panel job file produced by build_prompts. Training uses all
eligible rows outside the recipient's outer fold; output includes only rows
whose evaluation_cohort is true. Saved aggregate fit audits can be supplied
with --penalties to reuse selected penalties and final seeds without tuning.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from survey_audit import reference


JOB_FIELDS = (
    "respondent_id", "dataset", "target", "condition", "fold", "weight",
    "actual_option", "options", "side_codes", "profile", "profile_codes",
    "displayed_profile", "profile_fields", "history", "evaluation_cohort",
    "full_panel_profile_order", "donor_id", "question_type", "history_length",
    "requested_history_length", "source_respondent_id",
    "analysis_profile", "displayed_profile_codes", "displayed_analysis_profile",
    "donor_analysis_profile", "option_labels",
)


def read_jobs(path, conditions, target=None):
    """Read reference-relevant fields and discard language-model prompt text."""
    groups = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            job = json.loads(line)
            if job["condition"] not in conditions or (target and job["target"] != target):
                continue
            compact = {field: job[field] for field in JOB_FIELDS if field in job}
            groups[job["dataset"], job["target"], job["condition"]].append(compact)
    if not groups:
        raise ValueError("No jobs matched the requested reference conditions")
    return groups


def validate_cells(jobs):
    """Require unique respondents, fixed support, and positive survey weights."""
    identifiers = [str(job["respondent_id"]) for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Repeated respondent in one target and condition")
    support = tuple(str(code) for code in jobs[0]["options"])
    for job in jobs:
        if tuple(str(code) for code in job["options"]) != support:
            raise ValueError("Answer support changes within a target")
        if job["fold"] not in range(reference.OUTER_FOLDS):
            raise ValueError("Outer fold must be in 0 through 4")
        if not np.isfinite(job["weight"]) or job["weight"] <= 0:
            raise ValueError("Reference fitting requires a positive survey weight")
    return support


def prediction_records(jobs, probabilities, training_condition):
    """Keep analysis metadata alongside each held-out probability vector."""
    for job, values in zip(jobs, probabilities, strict=True):
        if not job.get("evaluation_cohort", True):
            continue
        record = dict(job)
        record["model"] = "reference"
        record["reference_training_condition"] = training_condition
        record["probabilities"] = {
            str(code): float(value) for code, value in zip(job["options"], values, strict=True)}
        yield record


def fit_target(jobs, item_order, target_number, saved_audits=None,
               shuffled_jobs=None, field_subset=None):
    """Yield five held-out fits, with optional retained Own-to-Shuffled transfer."""
    support = validate_cells(jobs)
    dataset, target, condition = (jobs[0][field] for field in ("dataset", "target", "condition"))
    fields = reference.selected_fields(jobs, dataset)
    if field_subset is not None:
        if not set(field_subset) <= set(fields):
            raise ValueError("Requested fields were not supplied to these jobs")
        fields = tuple(field for field in fields if field in field_subset)
    rows, continuous = reference.feature_rows(jobs, item_order, fields)
    transfer_by_person = None
    if shuffled_jobs is not None:
        if validate_cells(shuffled_jobs) != support:
            raise ValueError("Own and Shuffled answer support differ")
        transfer_by_person = {str(job["respondent_id"]): job for job in shuffled_jobs}
        expected = {str(job["respondent_id"]) for job in jobs if job.get("evaluation_cohort", True)}
        if not expected <= set(transfer_by_person):
            raise ValueError("Shuffled transfer is missing evaluation recipients")
        for job in jobs:
            identifier = str(job["respondent_id"])
            if identifier not in transfer_by_person:
                continue
            assigned = transfer_by_person[identifier]
            if any(job[field] != assigned[field] for field in
                   ("fold", "history", "actual_option", "weight", "profile")):
                raise ValueError("Shuffled inputs changed recipient metadata or history")
    for fold in range(reference.OUTER_FOLDS):
        training = [row for row in rows if row.job["fold"] != fold]
        prediction = [row for row in rows if row.job["fold"] == fold]
        if not training or not prediction:
            raise ValueError(f"Empty outer fold for {dataset}/{target}: {fold}")
        training_ids = {row.respondent_identifier for row in training}
        if training_ids & {row.respondent_identifier for row in prediction}:
            raise ValueError("Respondents overlap between training and prediction")
        maximum_iterations = 2000 if dataset == "issp" else 10000
        seed = reference.fit_seed(target_number, fold, 3)
        losses = {}
        inner_iterations = 0
        if saved_audits is not None:
            audit = saved_audits[(dataset, target, condition, fold)]
            if (audit["seed"] != seed or audit["fields"] != list(fields)
                    or audit["item_order"] != list(item_order)):
                raise ValueError("Saved penalty audit uses a different seed or feature layout")
            strength = float(audit["selected_inverse_regularization"])
            maximum_iterations = int(audit["maximum_iteration_count"])
            if strength not in reference.PENALTY_GRID:
                raise ValueError("Saved penalty is outside the frozen candidate grid")
        else:
            strength, losses, inner_iterations = reference.select_strength(
                training, continuous, len(support), target_number, fold, maximum_iterations)
        state, probabilities, iterations = reference.fit_retained(
            training, prediction, continuous, len(support), strength, seed, maximum_iterations)
        records = list(prediction_records([row.job for row in prediction], probabilities, condition))
        if transfer_by_person is not None:
            transfer = [transfer_by_person[row.respondent_identifier] for row in prediction
                        if row.job.get("evaluation_cohort", True)]
            if transfer:
                transfer_rows, _ = reference.feature_rows(transfer, item_order, fields)
                transferred = reference.predict_retained(state, transfer_rows)
                records.extend(prediction_records(transfer, transferred, condition))
        audit = {
            "dataset": dataset, "target": target, "condition": condition, "fold": fold,
            "selected_inverse_regularization": strength, "seed": seed,
            "maximum_iteration_count": maximum_iterations,
            "maximum_observed_iterations": max(iterations, inner_iterations),
            "inner_validation_losses": losses, "selected_penalty_reused": saved_audits is not None,
            "training_count": len(training), "held_out_count": len(prediction),
            "training_prediction_overlap": 0, "fields": list(fields),
            "item_order": list(item_order), "continuous_positions": list(continuous),
            "transferred_to_shuffled": transfer_by_person is not None,
        }
        yield records, audit


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True, help="Full-panel jobs JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Held-out predictions JSONL")
    parser.add_argument("--audit", type=Path, required=True, help="Aggregate fitting audit JSON")
    parser.add_argument("--conditions", nargs="+", default=["History", "Own"])
    parser.add_argument("--target", help="Optionally fit one canonical target")
    parser.add_argument("--fields", nargs="*", help="Optional demographic feature subset")
    parser.add_argument("--penalties", type=Path, help="Reuse a previous --audit file without tuning")
    parser.add_argument("--transfer-shuffled", action="store_true",
                        help="Apply retained Own fits to Shuffled jobs in the same input")
    parser.add_argument("--own-condition", default="Own")
    parser.add_argument("--shuffled-condition", default="Shuffled")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    from survey_audit.survey_specifications import TARGETS

    conditions = list(arguments.conditions)
    if arguments.transfer_shuffled:
        if arguments.own_condition not in conditions:
            raise ValueError("Transfer requires the Own condition among --conditions")
        if arguments.shuffled_condition in conditions:
            raise ValueError("Transfer uses Own fits; remove Shuffled from --conditions")
    inputs = conditions + ([arguments.shuffled_condition] if arguments.transfer_shuffled else [])
    groups = read_jobs(arguments.jobs, inputs, arguments.target)
    saved = None
    if arguments.penalties:
        rows = json.loads(arguments.penalties.read_text(encoding="utf-8"))["fits"]
        saved = {(row["dataset"], row["target"], row["condition"], row["fold"]): row for row in rows}
    audits = []
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.audit.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as output, threadpool_limits(limits=1):
        for (dataset, target, condition), jobs in groups.items():
            if condition not in conditions:
                continue
            item_order = tuple(TARGETS[dataset])
            target_number = int(target[1:]) if dataset == "issp" else item_order.index(target) + 1
            transferred = None
            if arguments.transfer_shuffled and condition == arguments.own_condition:
                transferred = groups[(dataset, target, arguments.shuffled_condition)]
            for records, audit in fit_target(jobs, item_order, target_number, saved,
                                             transferred, arguments.fields):
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                audits.append(audit)
                print(f"{dataset} {target} {condition}: fold {audit['fold']} complete", flush=True)
    result = {"outer_folds": 5, "inner_folds": 3, "penalty_grid": list(reference.PENALTY_GRID),
              "solver": "lbfgs", "tolerance": reference.TOLERANCE,
              "survey_weights": "WEIGHT_COM for ISSP; target-specific source weight for Pew",
              "fits": audits}
    arguments.audit.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
