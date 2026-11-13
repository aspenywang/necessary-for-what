"""Compute four fidelity errors and paired gains from saved probability vectors.

Individual Brier and population TVD average targets equally. Subgroup TVD
averages dimensions, targets within dimension, and categories within target
in that order. Contrast error averages the fixed human-supported pairs.
Basic 95% intervals are the paper's selected error-interval construction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from survey_audit.analysis_common import (
    BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, deletion_counts, human_pairs,
    interval, load_panels, means, subgroup_weights, write_csv,
)

METRICS = ('individual_brier', 'population_tvd_pp', 'subgroup_tvd_pp',
           'supported_pair_contrast_error_pp')


def metric_arrays(panel, counts, registry, survey_weighted=False):
    """Evaluate fixed predictions on shared resampling multiplicities."""
    row_count, model_count = len(counts), len(panel['labels'])
    output = {metric: np.zeros((row_count, model_count)) for metric in METRICS}
    group_weights = subgroup_weights(panel)
    supported = {key for key, row in registry.items() if row['human_class'] == 'human_supported'}
    for target, item in panel['items'].items():
        residuals = item['probabilities'] - item['human'][:, None, :]
        weights = item['weights'] if survey_weighted else None
        output['individual_brier'] += means(
            np.square(residuals).sum(axis=2), item['positions'], counts, weights)
        flattened = residuals.reshape(len(residuals), -1)
        population = means(flattened, item['positions'], counts, weights).reshape(
            row_count, model_count, -1)
        output['population_tvd_pp'] += 50 * np.abs(population).sum(axis=2)
        directional = {}
        for (field, category), members in item['groups'].items():
            distribution = means(flattened[members], item['positions'][members], counts,
                                 weights[members] if weights is not None else None).reshape(
                                     row_count, model_count, -1)
            output['subgroup_tvd_pp'] += group_weights[field, category, target] * 50 * np.abs(distribution).sum(axis=2)
            if item['side']:
                directional[field, category] = distribution[:, :, item['side']].sum(axis=2)
        for question, field, first, second in supported:
            if question == target:
                output['supported_pair_contrast_error_pp'] += 100 * np.abs(
                    directional[field, first] - directional[field, second])
    output['individual_brier'] /= len(panel['items'])
    output['population_tvd_pp'] /= len(panel['items'])
    if not group_weights:
        output['subgroup_tvd_pp'][:] = np.nan
    if supported:
        output['supported_pair_contrast_error_pp'] /= len(supported)
    else:
        output['supported_pair_contrast_error_pp'][:] = np.nan
    return output


def comparison_pairs(conditions):
    pairs = {(condition, 'History') for condition in conditions if condition != 'History' and 'History' in conditions}
    candidates = [('Own', 'Shuffled'), ('Own_10', 'Own'), ('Own_7', 'Own'),
                  ('Own_7', 'Own_10'), ('Own_4', 'Own'), ('Own_4', 'Own_7'),
                  ('Shuffled_10', 'Own_10'), ('ACS_10', 'Own_10'),
                  ('Nemotron_4', 'Own_4'), ('Nemotron_all', 'Own_4'),
                  ('Nemotron_all', 'Own'), ('Own_political', 'Own')]
    pairs.update((condition, comparator) for condition, comparator in candidates
                 if condition in conditions and comparator in conditions)
    return sorted(pairs)


def summarize_panel(panel, registry, method='basic', survey_weighted=False, universe='all'):
    identity = np.ones((1, len(panel['identifiers'])))
    arrays = metric_arrays(panel, np.vstack((panel['counts'], identity)), registry, survey_weighted)
    deletions = metric_arrays(panel, deletion_counts(panel['folds']), registry, survey_weighted) if method == 'bca' else None
    rows = []
    labels = panel['labels']
    for metric, values in arrays.items():
        for model in sorted({label[0] for label in labels}):
            conditions = [condition for name, condition in labels if name == model]
            specifications = [(condition, '') for condition in conditions] + comparison_pairs(conditions)
            for condition, comparator in specifications:
                column = labels.index((model, condition))
                estimates = values[:, column]
                deleted = deletions[metric][:, column] if deletions is not None else None
                if comparator:
                    baseline = labels.index((model, comparator))
                    estimates = values[:, baseline] - estimates
                    if deleted is not None:
                        deleted = deletions[metric][:, baseline] - deleted
                point = float(estimates[-1])
                lower, upper = (interval(point, estimates[:-1], method, deleted, panel['folds'])
                                if np.isfinite(estimates).all() else (np.nan, np.nan))
                rows.append(dict(dataset=panel['dataset'], model=model, condition=condition,
                    comparator=comparator, universe=universe,
                    history_length=';'.join(map(str, sorted({length for item in panel['items'].values()
                                               for length in item['history_lengths'][model, condition]}))), quantity='gain' if comparator else 'level', metric=metric,
                    point=point, bootstrap_mean=float(np.mean(estimates[:-1])),
                    bias_corrected_point=2 * point - float(np.mean(estimates[:-1])),
                    ci_low=lower, ci_high=upper, interval_method=method,
                    verdict=('improves' if lower > 0 else 'worsens' if upper < 0 else 'unresolved') if comparator else '',
                    supported_pairs=sum(row['human_class'] == 'human_supported' for row in registry.values()),
                    weight_basis='survey_weights' if survey_weighted else 'unit_weights',
                    bootstrap_repetitions=len(panel['counts']), bootstrap_seed=BOOTSTRAP_SEED,
                    notes='fixed_predictions;fixed_questions;paired_original_fold_resampling;gain=comparator_error-condition_error'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', nargs='+', required=True, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', choices=('basic', 'percentile', 'bca'), default='basic')
    parser.add_argument('--bootstrap-repetitions', type=int, default=BOOTSTRAP_REPETITIONS)
    parser.add_argument('--universe', choices=('all', 'prompted'), default='all',
                        help='Support pool: all analyst dimensions or the prompted fifteen.')
    parser.add_argument('--survey-weighted', action='store_true', help='Secondary weighting; current paper uses unit weights.')
    arguments = parser.parse_args()
    rows = []
    for panel in load_panels(arguments.predictions, arguments.bootstrap_repetitions):
        registry = human_pairs(panel, arguments.survey_weighted, arguments.universe)
        rows.extend(summarize_panel(panel, registry, arguments.interval, arguments.survey_weighted, arguments.universe))
    write_csv(arguments.output, rows)


if __name__ == '__main__':
    main()
