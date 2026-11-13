"""Classify supported group contrasts and regroup by the profile shown.

Human support requires absolute contrast at least five percentage points and
its percentile 95% interval excluding zero. A strict null has its percentile
90% interval entirely inside (-5, 5). Synthetic support uses the same five-point
and percentile-95 rule. On human-supported pairs, supported same-sign contrasts
are reproduced, supported opposite-sign contrasts reversed, and others missed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from survey_audit.analysis_common import (
    BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, human_pairs, load_panels, means, write_csv,
)


def classification(human_class, human, point, lower, upper):
    supported = abs(point) >= 5 and (lower > 0 or upper < 0)
    if human_class == 'human_supported':
        if not supported:
            return 'missed'
        return 'reproduced' if np.sign(point) == np.sign(human) else 'reversed'
    if human_class == 'human_null':
        return 'unsupported' if supported else 'no_supported_difference'
    return 'indeterminate'


def contrast_rows(panel, registry, grouping='both', survey_weighted=False,
                  shuffled_condition='Shuffled', universe='all'):
    """Freeze human own-profile support; retain bootstrap draws on recipients.

    The History placebo uses the same shown Shuffled profiles. Only fields in
    those shown profiles enter shown-group contrasts. Own-group rows retain
    every eligible analyst dimension, including fields absent from prompts.
    """
    counts = np.vstack((panel['counts'], np.ones((1, len(panel['identifiers'])))))
    requested = ('own', 'shown') if grouping == 'both' else (grouping,)
    output = []
    for target, item in panel['items'].items():
        keys = [key for key in registry if key[0] == target]
        if not keys:
            continue
        values = item['probabilities'][:, :, item['side']].sum(axis=2)
        for column, (model, condition) in enumerate(panel['labels']):
            for basis in requested:
                if basis == 'own':
                    profiles = item['profiles']
                else:
                    shown_label = (model, shuffled_condition) if condition == 'History' else (model, condition)
                    if shown_label not in item['displayed']:
                        continue
                    profiles = item['displayed'][shown_label]
                fields = {field for profile in profiles for field in profile}
                required = {(field, category) for _, field, first, second in keys if field in fields
                            for category in (first, second)}
                estimates = {}
                sizes = {}
                for field, category in sorted(required):
                    members = np.asarray([index for index, profile in enumerate(profiles)
                                          if profile.get(field) == category], dtype=int)
                    if not len(members):
                        continue
                    estimates[field, category] = 100 * means(
                        values[members, column:column + 1], item['positions'][members], counts,
                        item['weights'][members] if survey_weighted else None)[:, 0]
                    sizes[field, category] = len(members)
                for key in keys:
                    _, field, first, second = key
                    if field not in fields:
                        continue
                    if (field, first) not in estimates or (field, second) not in estimates:
                        raise ValueError('A frozen pair has an empty displayed-profile group')
                    draws = estimates[field, first] - estimates[field, second]
                    lower, upper = np.percentile(draws[:-1], [2.5, 97.5])
                    point = float(draws[-1])
                    lower90, upper90 = np.percentile(draws[:-1], [5, 95])
                    human = registry[key]
                    human_point = human['human_contrast_pp']
                    supported = human['human_class'] == 'human_supported'
                    output.append(dict(dataset=panel['dataset'], model=model, condition=condition,
                        grouping=basis, universe=universe, target=target, attribute=field,
                        history_length=';'.join(map(str, item['history_lengths'][model, condition])), first_group=first,
                        second_group=second, **human, synthetic_contrast_pp=point,
                        synthetic_ci_low=float(lower), synthetic_ci_high=float(upper),
                        synthetic_first_share_pp=float(estimates[field, first][-1]),
                        synthetic_second_share_pp=float(estimates[field, second][-1]),
                        synthetic_ci90_low=float(lower90), synthetic_ci90_high=float(upper90),
                        synthetic_null=bool(lower90 > -5 and upper90 < 5),
                        classification=classification(human['human_class'], human_point, point, lower, upper),
                        signed_ratio=point / human_point if supported else np.nan,
                        synthetic_first_group_count=sizes[field, first],
                        synthetic_second_group_count=sizes[field, second],
                        weight_basis='survey_weights' if survey_weighted else 'unit_weights',
                        bootstrap_repetitions=len(panel['counts']), bootstrap_seed=BOOTSTRAP_SEED,
                        notes='human_support_frozen_on_own_profile;percentile_intervals;recipient_resampling'))
    return output


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = row['dataset'], row['model'], row['condition'], row['grouping']
        groups[key + ('all',)].append(row)
        groups[key + (row['attribute'],)].append(row)
    output = []
    for (dataset, model, condition, grouping, attribute), selected in sorted(groups.items()):
        supported = [row for row in selected if row['human_class'] == 'human_supported']
        null = [row for row in selected if row['human_class'] == 'human_null']
        counts = {name: sum(row['classification'] == name for row in supported)
                  for name in ('reproduced', 'missed', 'reversed')}
        ratios = np.asarray([row['signed_ratio'] for row in supported])
        null_missed = sum(row['classification'] == 'missed' and row['synthetic_null'] for row in supported)
        same_sign = int((ratios > 0).sum())
        attenuated = int(((ratios > 0) & (ratios < 1)).sum())
        amplified = int((ratios > 1).sum())
        output.append(dict(dataset=dataset, model=model, condition=condition,
            grouping=grouping, universe=selected[0]['universe'],
            history_length=selected[0]['history_length'], attribute=attribute, supported_pairs=len(supported),
            human_null_pairs=len(null), indeterminate_pairs=len(selected) - len(supported) - len(null),
            reproduced_count=counts['reproduced'], missed_count=counts['missed'], reversed_count=counts['reversed'],
            reproduced_share=counts['reproduced'] / len(supported) if supported else np.nan,
            missed_share=counts['missed'] / len(supported) if supported else np.nan,
            reversed_share=counts['reversed'] / len(supported) if supported else np.nan,
            unsupported_count=sum(row['classification'] == 'unsupported' for row in null),
            median_signed_ratio=float(np.median(ratios)) if len(ratios) else np.nan,
            synthetic_null_missed_count=null_missed, uncertain_missed_count=counts['missed'] - null_missed,
            median_same_sign_ratio=float(np.median(ratios[ratios > 0])) if same_sign else np.nan,
            same_sign_pairs=same_sign, attenuated_pairs=attenuated, amplified_pairs=amplified,
            conditional_attenuation_share=attenuated / same_sign if same_sign else np.nan,
            notes='pair_pooled;attenuation=number(0<ratio<1)/number(ratio>0);no_synthetic_support_filter_for_ratio'))
    return output


def transition_rows(rows):
    keyed = {(row['dataset'], row['model'], row['condition'], row['grouping'], row['target'],
              row['attribute'], row['first_group'], row['second_group']): row for row in rows}
    counts = defaultdict(int)
    for key, row in keyed.items():
        dataset, model, condition, grouping, target, attribute, first, second = key
        if condition == 'History' or row['human_class'] != 'human_supported':
            continue
        previous = keyed.get((dataset, model, 'History', grouping, target, attribute, first, second))
        if previous is None:
            continue
        for dimension in ('all', attribute):
            counts[dataset, model, condition, grouping, dimension,
                   previous['classification'], row['classification']] += 1
    return [dict(zip(('dataset', 'model', 'condition', 'grouping', 'attribute', 'history_class', 'condition_class'), key),
                 pair_count=value, universe=rows[0]['universe'], history_length=rows[0]['history_length'])
            for key, value in sorted(counts.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--grouping', choices=('own', 'shown', 'both'), default='both')
    parser.add_argument('--shuffled-condition', default='Shuffled', help='Shown profiles for the History placebo.')
    parser.add_argument('--bootstrap-repetitions', type=int, default=BOOTSTRAP_REPETITIONS)
    parser.add_argument('--survey-weighted', action='store_true')
    parser.add_argument('--universe', choices=('all', 'prompted'), default='all',
                        help='Prompted excludes the two Pew political analyst dimensions.')
    arguments = parser.parse_args()
    rows = []
    for panel in load_panels(arguments.predictions, arguments.bootstrap_repetitions):
        registry = human_pairs(panel, arguments.survey_weighted, arguments.universe)
        rows.extend(contrast_rows(panel, registry, arguments.grouping,
                                  arguments.survey_weighted, arguments.shuffled_condition, arguments.universe))
    write_csv(arguments.output_dir / 'contrast_results.csv', rows)
    write_csv(arguments.output_dir / 'contrast_summary_results.csv', summarize(rows))
    transitions = transition_rows(rows)
    if transitions:
        write_csv(arguments.output_dir / 'contrast_transition_results.csv', transitions)


if __name__ == '__main__':
    main()
