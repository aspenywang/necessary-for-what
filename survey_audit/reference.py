"""Weighted nested cross-fitted multinomial reference and retained prediction.

Five outer folds are supplied by data preparation. Three inner folds keep
identical demographic signatures together. The inverse penalty grid is
0.03, 0.3, 3.0; held-out survey-weighted log loss chooses the penalty. Numeric
ISSP age is standardized and all remaining features use one-hot encoding.
"""

from collections import defaultdict
import math
import random

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PENALTY_GRID = (0.03, 0.3, 3.0)
OUTER_FOLDS = 5
INNER_FOLDS = 3
TOLERANCE = 1e-6
PROBABILITY_FLOOR = 1e-15
NOT_SUPPLIED = "not_supplied"
ISSP_ORIGINAL_FIELDS = ("AGE", "SEX", "EDULEVEL", "WORK", "RELIGGRP", "ATTEND", "URBRURAL")
ISSP_ADDED_FIELDS = ("US_INC", "US_ETHN1", "US_ETHN2", "MARITAL", "US_REG", "v60", "HOMPOP", "ISCO08")
PEW_FIELDS = ("CREGION", "F_METRO", "SEX", "AGE", "EDUCATION", "RACE", "CITIZEN", "F_NATIVITY", "MARITAL", "F_PARENT", "F_E3", "INCOME", "RELIG", "RELIGATTEND", "F_BORN")
PEW_POLITICAL_FIELDS = ("POLPARTY", "POLIDEOLOGY", "F_REG")
NATIVE_FIELDS = ("sex", "age", "marital_status", "education_level", "bachelors_field",
                 "occupation", "city", "state", "zipcode")


class FeatureRow:
    """One target answer, its supplied features, and its survey weight."""

    def __init__(self, job, feature_values, target_position):
        self.job = job
        self.feature_values = tuple(feature_values)
        self.target_position = int(target_position)
        self.respondent_identifier = str(job["respondent_id"])
        self.analysis_weight = float(job["weight"])


def exact_demographic_groups(signatures):
    """Group identical complete signatures with the original stable ordering."""
    grouped = defaultdict(list)
    for identifier, signature in signatures.items():
        grouped[tuple(signature)].append(str(identifier))
    return tuple(tuple(sorted(identifiers)) for _, identifiers in
                 sorted(grouped.items(), key=lambda pair: pair[0]))


def pair_aware_folds(exact_groups, cell_counts, fold_count=5, random_seed=20260904):
    """Assign demographic groups atomically, balancing people then target cells."""
    groups = [tuple(sorted(group)) for group in exact_groups]
    identifiers = [identifier for group in groups for identifier in group]
    if (fold_count < 2 or not identifiers or len(identifiers) != len(set(identifiers))
            or set(identifiers) != set(cell_counts)):
        raise ValueError("Groups must partition the respondents")
    random.Random(random_seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    people = [0] * fold_count
    cells = [0] * fold_count
    group_counts = [0] * fold_count
    assigned = {}
    for group in groups:
        fold = min(range(fold_count), key=lambda value:
                   (people[value], cells[value], group_counts[value], value))
        assigned.update((identifier, fold) for identifier in group)
        people[fold] += len(group)
        cells[fold] += sum(int(cell_counts[identifier]) for identifier in group)
        group_counts[fold] += 1
    return assigned


def fit_seed(target_number, outer_fold, method_offset):
    """Preserve the original history-condition seed for every reference fit."""
    return 42 + 100 * int(target_number) + 10 * int(outer_fold) + 10000 + method_offset


def selected_fields(jobs, dataset):
    """Keep canonical field order and include missing added-feature columns."""
    if dataset == "pew" and jobs[0]["condition"] == "Nemotron_all":
        return NATIVE_FIELDS
    fields = set()
    for job in jobs:
        fields.update(job.get("profile_fields", job.get("displayed_profile", {}).keys()))
    fields.discard("Country")
    canonical = (ISSP_ORIGINAL_FIELDS + ISSP_ADDED_FIELDS if dataset == "issp"
                 else PEW_FIELDS + PEW_POLITICAL_FIELDS)
    unknown = fields - set(canonical)
    if unknown:
        raise ValueError(f"Unrecognized reference fields: {sorted(unknown)}")
    return tuple(field for field in canonical if field in fields)


def feature_rows(jobs, item_order, fields):
    """Encode recipient history with displayed demographics, without the target.

    ISSP order is numeric age, country and original categorical fields, all
    history columns, then the eight additional categorical fields. Pew uses
    all categorical displayed fields followed by all history columns.
    """
    rows = []
    for job in jobs:
        supplied = {entry["target"]: str(entry["answer"]) for entry in job["history"]}
        if (len(supplied) != len(job["history"]) or job["target"] in supplied
                or not set(supplied) <= set(item_order)):
            raise ValueError("Invalid history support or target leakage")
        history = tuple("response_code_" + supplied[item] if item in supplied
                        else NOT_SUPPLIED for item in item_order)
        displayed = job.get("displayed_profile", {})

        def category(field):
            value = displayed.get(field)
            return NOT_SUPPLIED if value is None else str(value)

        if job["dataset"] == "issp":
            values = []
            if "AGE" in fields:
                values.append(float(displayed["AGE"]))
            if fields:
                values.append(str(job["profile"].get("Country", "United States")))
            values.extend(category(field) for field in ISSP_ORIGINAL_FIELDS[1:]
                          if field in fields)
            values.extend(history)
            values.extend(category(field) for field in ISSP_ADDED_FIELDS if field in fields)
        else:
            values = [category(field) for field in fields] + list(history)
        codes = [str(code) for code in job["options"]]
        rows.append(FeatureRow(job, values, codes.index(str(job["actual_option"]))))
    continuous = (0,) if jobs[0]["dataset"] == "issp" and "AGE" in fields else ()
    return rows, continuous


def inner_fold_assignments(rows, target_number, outer_fold):
    """Preserve true-profile groups and the original ISSP Shuffled grouping."""
    signatures = {}
    for row in rows:
        profile = row.job["profile"]
        if row.job["dataset"] == "issp":
            if row.job["condition"].startswith("Shuffled"):
                profile = {**profile, **row.job["displayed_profile"]}
            signature = (str(profile.get("Country", "United States")),) + tuple(
                str(profile[field]) for field in ISSP_ORIGINAL_FIELDS)
        else:
            signature = tuple(str(profile[field]) for field in PEW_FIELDS)
        identifier = row.respondent_identifier
        if identifier in signatures and signatures[identifier] != signature:
            raise ValueError("Respondent demographics change across target rows")
        signatures[identifier] = signature
    return pair_aware_folds(exact_demographic_groups(signatures),
                           dict.fromkeys(signatures, 1), INNER_FOLDS,
                           fit_seed(target_number, outer_fold, 1))


def fit_transformer(training, prediction, continuous):
    """Fit preprocessing only on training rows and retain the transformer."""
    if not training or not prediction:
        raise ValueError("Nonempty training and prediction rows are required")
    width = len(training[0].feature_values)
    if any(len(row.feature_values) != width for row in training + prediction):
        raise ValueError("Feature widths differ")
    categorical = [position for position in range(width) if position not in continuous]
    transformers = []
    if continuous:
        transformers.append(("continuous_features", StandardScaler(), list(continuous)))
    if categorical:
        transformers.append(("categorical_features", OneHotEncoder(
            handle_unknown="ignore", sparse_output=True), categorical))
    transformer = ColumnTransformer(transformers=tuple(transformers), sparse_threshold=1.0)
    training_matrix = transformer.fit_transform([row.feature_values for row in training])
    prediction_matrix = transformer.transform([row.feature_values for row in prediction])
    return transformer, training_matrix, prediction_matrix


def align_probabilities(classes, probabilities, option_count):
    """Restore all answer columns, including classes absent from training."""
    aligned = np.zeros((len(probabilities), option_count), dtype=np.float64)
    for column, position in enumerate(classes):
        aligned[:, int(position)] = probabilities[:, column]
    return aligned / aligned.sum(axis=1, keepdims=True)


def fit_matrix(training_matrix, training, prediction_matrix, option_count,
               strength, random_seed, maximum_iterations):
    """Fit the original weighted L-BFGS multinomial estimator."""
    observed = sorted({row.target_position for row in training})
    if len(observed) == 1:
        constant = np.zeros(option_count)
        constant[observed[0]] = 1.0
        return None, np.tile(constant, (prediction_matrix.shape[0], 1)), 0
    classifier = LogisticRegression(C=strength, solver="lbfgs", tol=TOLERANCE,
                                    max_iter=maximum_iterations, random_state=random_seed)
    classifier.fit(training_matrix,
                   np.asarray([row.target_position for row in training]),
                   sample_weight=np.asarray([row.analysis_weight for row in training]))
    iterations = int(np.max(classifier.n_iter_))
    if iterations >= maximum_iterations:
        raise RuntimeError(f"Reference fit reached its iteration limit: {iterations}")
    probabilities = align_probabilities(classifier.classes_,
                                        classifier.predict_proba(prediction_matrix), option_count)
    return classifier, probabilities, iterations


def weighted_log_loss(rows, probabilities):
    """Survey-weighted target log loss used solely to choose the penalty."""
    weight = sum(row.analysis_weight for row in rows)
    return sum(-row.analysis_weight * math.log(max(
        float(probabilities[index, row.target_position]), PROBABILITY_FLOOR))
               for index, row in enumerate(rows)) / weight


def select_strength(training, continuous, option_count, target_number,
                    outer_fold, maximum_iterations):
    """Three-fold weighted log-loss tuning; ties favor the smaller strength."""
    assignments = inner_fold_assignments(training, target_number, outer_fold)
    losses = dict.fromkeys(PENALTY_GRID, 0.0)
    total_weight = 0.0
    maximum_observed = 0
    for inner_fold in range(INNER_FOLDS):
        inner_training = [row for row in training if assignments[row.respondent_identifier] != inner_fold]
        validation = [row for row in training if assignments[row.respondent_identifier] == inner_fold]
        _, training_matrix, prediction_matrix = fit_transformer(inner_training, validation, continuous)
        weight = sum(row.analysis_weight for row in validation)
        total_weight += weight
        for strength in PENALTY_GRID:
            _, probabilities, iterations = fit_matrix(
                training_matrix, inner_training, prediction_matrix, option_count, strength,
                fit_seed(target_number, outer_fold, 2), maximum_iterations)
            losses[strength] += weight * weighted_log_loss(validation, probabilities)
            maximum_observed = max(maximum_observed, iterations)
    losses = {strength: value / total_weight for strength, value in losses.items()}
    return min(losses, key=lambda strength: (losses[strength], strength)), losses, maximum_observed


def fit_retained(training, prediction, continuous, option_count, strength,
                 random_seed, maximum_iterations):
    """Retain the Own transformer and estimator for subsequent Shuffled input."""
    transformer, training_matrix, prediction_matrix = fit_transformer(training, prediction, continuous)
    classifier, probabilities, iterations = fit_matrix(
        training_matrix, training, prediction_matrix, option_count, strength,
        random_seed, maximum_iterations)
    state = {"transformer": transformer, "classifier": classifier,
             "option_count": option_count,
             "constant_probability": probabilities[0] if classifier is None else None}
    return state, probabilities, iterations


def predict_retained(state, rows):
    """Apply an already fitted Own object without fitting on Shuffled inputs."""
    matrix = state["transformer"].transform([row.feature_values for row in rows])
    classifier = state["classifier"]
    if classifier is None:
        return np.tile(state["constant_probability"], (len(rows), 1))
    return align_probabilities(classifier.classes_, classifier.predict_proba(matrix),
                               state["option_count"])
