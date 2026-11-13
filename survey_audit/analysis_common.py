"""Shared fixed-prediction estimators and original-fold respondent bootstrap.

Adapted from the original shared bootstrap, group registry, Section 4 endpoint,
and fold-centered interval implementations. Group eligibility uses original
survey weights; current-paper estimates use unit respondent weights.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

BOOTSTRAP_SEED = 20260910
BOOTSTRAP_REPETITIONS = 10000
ISSP_FIELDS = {
    'AGE': 'age_group', 'SEX': 'sex', 'EDULEVEL': 'education',
    'WORK': 'work_status', 'RELIGGRP': 'religious_affiliation',
    'ATTEND': 'religious_attendance', 'URBRURAL': 'urban_rural_residence',
    'US_INC': 'household_income', 'US_ETHN1': 'ethnic_origin_first_mention',
    'US_ETHN2': 'ethnic_origin_second_mention', 'MARITAL': 'marital_status',
    'US_REG': 'region', 'v60': 'citizenship', 'HOMPOP': 'household_size',
    'ISCO08': 'occupation_major_group',
}
EXCLUDED_CATEGORIES = {'', 'Refused', 'Not recorded'}
PEW_FIELDS = ('CREGION', 'F_METRO', 'SEX', 'AGE', 'EDUCATION', 'RACE',
              'CITIZEN', 'F_NATIVITY', 'MARITAL', 'F_PARENT', 'F_E3', 'INCOME',
              'RELIG', 'RELIGATTEND', 'F_BORN', 'POLPARTY', 'POLIDEOLOGY')


def iter_json_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_json_rows(path):
    return list(iter_json_rows(path))


def write_csv(path, rows):
    """Write an aggregate table, retaining undefined cells as empty strings."""
    if not rows:
        raise ValueError('No aggregate rows were produced')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: '' if isinstance(value, (float, np.floating))
                          and not np.isfinite(value) else value
                          for key, value in row.items()} for row in rows)


def age_group(value):
    age = int(float(value))
    if age <= 29:
        return '18_to_29'
    if age <= 44:
        return '30_to_44'
    if age <= 59:
        return '45_to_59'
    return '60_or_older'


def analysis_profile(row, displayed=False):
    """Use source codes for ISSP and literal displayed labels for Pew.

    Raw occupation unit codes are collapsed to major groups. Missing ISSP
    codes are omitted before grouping; age is collapsed to four fixed bins.
    """
    name = 'displayed_analysis_profile' if displayed else 'analysis_profile'
    fallback = 'displayed_profile' if displayed else 'profile'
    source = row.get(name, row.get(fallback, {}))
    if row['dataset'].lower() != 'issp':
        return {field: str(value) for field, value in source.items()
                if field in PEW_FIELDS and value is not None
                and str(value) not in EXCLUDED_CATEGORIES}
    result = {}
    for field, value in source.items():
        if value is None or str(value).strip() in EXCLUDED_CATEGORIES:
            continue
        if field in ISSP_FIELDS:
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0:
                continue
            category = (age_group(numeric) if field == 'AGE' else
                        str(int(numeric) // 1000 if numeric >= 10 else int(numeric))
                        if field == 'ISCO08' else str(int(numeric)))
            result[ISSP_FIELDS[field]] = category
        elif field in ISSP_FIELDS.values():
            result[field] = str(value)
    return result


def bootstrap_counts(folds, repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED):
    """Exact original multinomial bootstrap, stratified by fixed outer folds."""
    if repetitions < 1:
        raise ValueError('Bootstrap repetitions must be positive')
    identifiers = sorted(folds)
    folded = np.asarray([folds[identifier] for identifier in identifiers])
    result = np.zeros((repetitions, len(identifiers)), dtype=np.int16)
    generator = np.random.default_rng(seed)
    for fold in np.unique(folded):
        positions = np.flatnonzero(folded == fold)
        result[:, positions] = generator.multinomial(
            len(positions), np.full(len(positions), 1 / len(positions)),
            size=repetitions).astype(np.int16)
    return identifiers, result


def deletion_counts(folds):
    """Delete each recipient and renormalize remaining members of its fold."""
    folds = np.asarray(folds)
    result = np.ones((len(folds), len(folds)))
    for index in range(len(folds)):
        members = folds == folds[index]
        count = int(members.sum())
        if count < 2:
            raise ValueError('BCa requires at least two people in each fold')
        result[index, members] = count / (count - 1)
        result[index, index] = 0
    return result


def acceleration_from_deletions(values, folds):
    values, folds = np.asarray(values, dtype=float), np.asarray(folds)
    influences = np.empty_like(values)
    for fold in np.unique(folds):
        members = folds == fold
        count = int(members.sum())
        influences[members] = (count - 1) / count * (values[members].mean() - values[members])
    squared = float(np.square(influences).sum())
    return float(np.power(influences, 3).sum() / (6 * squared ** 1.5)) if squared else 0.


def interval(point, draws, method='basic', deletions=None, folds=None):
    """95% interval; percentile group classifications remain a separate rule."""
    draws = np.asarray(draws, dtype=float)
    if not np.isfinite(draws).all():
        raise ValueError('Undefined bootstrap statistic')
    lower, upper = np.percentile(draws, [2.5, 97.5])
    if method == 'basic':
        return float(2 * point - upper), float(2 * point - lower)
    if method == 'percentile':
        return float(lower), float(upper)
    if method != 'bca' or deletions is None or folds is None:
        raise ValueError('BCa requires fold-preserving deletion estimates')
    acceleration = acceleration_from_deletions(deletions, folds)
    fraction = float(((draws < point).sum() + .5 * (draws == point).sum()) / len(draws))
    if fraction < .01 or fraction > .99:
        return float('nan'), float('nan')
    bias = ndtri(fraction)
    locations = bias + ndtri(np.asarray([.025, .975]))
    denominator = 1 - acceleration * locations
    if np.any(denominator <= 0):
        return float('nan'), float('nan')
    return tuple(map(float, np.quantile(draws, ndtr(bias + locations / denominator))))


def means(values, positions, counts, weights=None):
    values = np.asarray(values, dtype=float)
    local = counts[:, positions].astype(float)
    if weights is not None:
        local *= np.asarray(weights)[None, :]
    denominator = local.sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError('An eligible group is empty in a bootstrap draw')
    return np.matmul(local, values) / denominator[:, None]


def load_panels(paths, repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED):
    """Align supplied model/condition predictions on identical human cells.

    Each line follows the scoring schema: dataset, respondent_id, target,
    condition, model, fold, weight, actual_option, options, side_codes, profile,
    displayed_profile and probabilities. Optional analysis_profile fields hold
    raw category codes independently of the text shown to the model.
    """
    by_dataset = defaultdict(dict)
    needed = ('respondent_id', 'target', 'model', 'condition', 'fold', 'weight',
              'actual_option', 'options', 'side_codes', 'profile', 'analysis_profile',
              'displayed_profile', 'displayed_analysis_profile', 'history', 'history_length',
              'requested_history_length', 'probabilities', 'dataset')
    for path in paths:
        for row in iter_json_rows(path):
            if not row.get('evaluation_cohort', row.get('eval_cohort', True)):
                continue
            identity = (str(row['respondent_id']), row['target'], row['model'], row['condition'])
            dataset = by_dataset[row['dataset']]
            if identity in dataset:
                raise ValueError('Duplicate prediction cell')
            dataset[identity] = {field: row[field] for field in needed if field in row}
    panels = []
    for name, records in sorted(by_dataset.items()):
        labels = sorted({key[2:] for key in records})
        canonical = {key[:2]: row for key, row in records.items() if key[2:] == labels[0]}
        cells = set(canonical)
        for label in labels:
            if {key[:2] for key in records if key[2:] == label} != cells:
                raise ValueError('Models and conditions must have identical evaluation cells')
        folds = {}
        for (person, _), row in canonical.items():
            if person in folds and folds[person] != row['fold']:
                raise ValueError('A respondent has inconsistent folds')
            folds[person] = row['fold']
        identifiers, draws = bootstrap_counts(folds, repetitions, seed)
        positions = {person: index for index, person in enumerate(identifiers)}
        items = {}
        for target in sorted({key[1] for key in cells}):
            people = sorted(person for person, question in cells if question == target)
            rows = [canonical[person, target] for person in people]
            options = list(map(str, rows[0]['options']))
            side = list(map(str, rows[0]['side_codes']))
            weights = np.asarray([float(row['weight']) for row in rows])
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError('Survey weights must be positive and finite')
            profiles = [analysis_profile(row) for row in rows]
            probabilities = np.empty((len(rows), len(labels), len(options)))
            displayed = {}
            for column, label in enumerate(labels):
                shown = []
                for index, person in enumerate(people):
                    row = records[(person, target) + label]
                    original = rows[index]
                    if (list(map(str, row['options'])) != options or
                        list(map(str, row['side_codes'])) != side or
                        str(row['actual_option']) != str(original['actual_option']) or
                        row['fold'] != original['fold'] or float(row['weight']) != weights[index] or
                        analysis_profile(row) != profiles[index]):
                        raise ValueError('Human metadata differ across paired predictions')
                    probabilities[index, column] = [float(row['probabilities'][code]) for code in options]
                    shown.append(analysis_profile(row, displayed=True))
                displayed[label] = shown
            if not np.isfinite(probabilities).all() or (probabilities < 0).any():
                raise ValueError('Invalid probability vector')
            np.testing.assert_allclose(probabilities.sum(axis=2), 1, atol=1e-7, rtol=0)
            groups = {}
            for field in sorted({field for profile in profiles for field in profile}):
                for category in sorted({profile[field] for profile in profiles if field in profile}):
                    members = np.asarray([index for index, profile in enumerate(profiles)
                                          if profile.get(field) == category], dtype=int)
                    group_weights = weights[members]
                    effective = group_weights.sum() ** 2 / np.square(group_weights).sum()
                    if len(members) >= 30 and effective >= 20:
                        groups[field, category] = members
            truth = np.asarray([options.index(str(row['actual_option'])) for row in rows])
            items[target] = dict(people=people, positions=np.asarray([positions[person] for person in people]),
                options=options, side=[options.index(code) for code in side], weights=weights,
                truth=truth, human=np.eye(len(options))[truth], profiles=profiles, displayed=displayed,
                groups=groups, probabilities=probabilities,
                history_lengths={label: sorted({str(records[(person, target) + label].get(
                    'requested_history_length', len(records[(person, target) + label].get('history', []))))
                    for person in people}) for label in labels})
        panels.append(dict(dataset=name, identifiers=identifiers, labels=labels, items=items,
                           folds=np.asarray([folds[person] for person in identifiers]), counts=draws))
    if not panels:
        raise ValueError('No prediction rows')
    return panels


def human_pairs(panel, survey_weighted=False, universe='all'):
    """Select support once from humans before any model comparison."""
    counts = np.vstack((panel['counts'], np.ones((1, len(panel['identifiers'])))))
    results = {}
    for target, item in panel['items'].items():
        if not item['side']:
            continue
        values = np.isin(item['truth'], item['side']).astype(float)[:, None]
        estimates = {key: 100 * means(values[members], item['positions'][members], counts,
                       item['weights'][members] if survey_weighted else None)[:, 0]
                     for key, members in item['groups'].items()}
        for field in sorted({key[0] for key in estimates}):
            if universe == 'prompted' and field in ('POLPARTY', 'POLIDEOLOGY'):
                continue
            categories = sorted(category for attribute, category in estimates if attribute == field)
            for first, second in combinations(categories, 2):
                contrast = estimates[field, first] - estimates[field, second]
                lower, upper = np.percentile(contrast[:-1], [2.5, 97.5])
                null_lower, null_upper = np.percentile(contrast[:-1], [5, 95])
                point = float(contrast[-1])
                supported = abs(point) >= 5 and (lower > 0 or upper < 0)
                is_null = null_lower > -5 and null_upper < 5
                results[target, field, first, second] = dict(
                    human_first_share_pp=float(estimates[field, first][-1]),
                    human_second_share_pp=float(estimates[field, second][-1]),
                    human_first_ci_low=float(np.percentile(estimates[field, first][:-1], 2.5)),
                    human_first_ci_high=float(np.percentile(estimates[field, first][:-1], 97.5)),
                    human_second_ci_low=float(np.percentile(estimates[field, second][:-1], 2.5)),
                    human_second_ci_high=float(np.percentile(estimates[field, second][:-1], 97.5)),
                    human_contrast_pp=point, human_ci_low=float(lower), human_ci_high=float(upper),
                    human_null_ci_low=float(null_lower), human_null_ci_high=float(null_upper),
                    human_class='human_supported' if supported else 'human_null' if is_null else 'indeterminate',
                    first_group_count=len(item['groups'][field, first]),
                    second_group_count=len(item['groups'][field, second]))
    return results


def subgroup_weights(panel):
    keys = [(field, category, target) for target, item in panel['items'].items()
            for field, category in item['groups']]
    group_counts = Counter((key[0], key[2]) for key in keys)
    target_counts = Counter(field for field, _ in group_counts)
    return {key: 1 / (len(target_counts) * target_counts[key[0]] * group_counts[key[0], key[2]])
            for key in keys}
