"""Construct paired profiles and circular target-specific observed histories."""

import argparse
import copy
import json
from pathlib import Path
import random

import numpy as np

from survey_audit.prepare_data import read_rows, write_rows
from survey_audit.survey_specifications import DATASET_FIELDS, FIELD_LABELS, PEW_POLITICAL_FIELDS

PEW_TEN = ('CREGION', 'SEX', 'AGE', 'EDUCATION', 'RACE', 'CITIZEN', 'F_NATIVITY', 'MARITAL', 'F_E3', 'INCOME')
PEW_SEVEN = ('CREGION', 'SEX', 'AGE', 'EDUCATION', 'RACE', 'CITIZEN', 'F_NATIVITY')
PEW_FOUR = ('CREGION', 'SEX', 'AGE', 'EDUCATION')
ISSP_FOUR = ('US_REG', 'SEX', 'AGE', 'EDULEVEL')
CONDITIONS = ('History', 'Own', 'Shuffled', 'Own_10', 'Shuffled_10', 'Own_7', 'Own_4', 'ACS_10', 'Nemotron_4', 'Nemotron_all', 'Own_political')
NATIVE_LABELS = {'sex': 'Sex', 'age': 'Age', 'marital_status': 'Marital status',
                 'education_level': 'Education level', 'bachelors_field': "Bachelor's field",
                 'occupation': 'Occupation', 'city': 'City', 'state': 'State', 'zipcode': 'ZIP code'}


def donor_mapping(rows, dataset, seed=20261017):
    """Move complete profiles once; Pew permits identities and ISSP rejects them."""
    if dataset == 'pew':
        ordered = sorted(rows, key=lambda row: row['full_panel_profile_order'])
        identifiers = [row['respondent_id'] for row in ordered]
        donors = identifiers.copy()
        random.Random(seed).shuffle(donors)
    else:
        identifiers = sorted(row['respondent_id'] for row in rows)
        generator = np.random.default_rng(seed)
        while True:
            donors = generator.permutation(identifiers).tolist()
            if all(person != donor for person, donor in zip(identifiers, donors)):
                break
    return dict(zip(identifiers, donors))


def select_history(instrument, answers, target, dataset, size=8):
    """Use the frozen circular rule, with the Pew family-diversity first pass."""
    if size == 0:
        return []
    order = list(instrument)
    position = order.index(target)
    priority = order[position + 1:] + order[:position]
    if dataset == 'pew':
        eligible = [item for item in priority if item in answers and instrument[item]['history_eligible']
                    and instrument[item]['family'] != instrument[target]['family']]
        size = len(eligible) if size == 'all' else min(size, len(eligible))
        if size == 0:
            return []
        selected, families = [], set()
        for item in eligible:
            family = instrument[item]['family']
            if family not in families:
                selected.append(item)
                families.add(family)
                if len(selected) == size:
                    return selected
        selected += [item for item in eligible if item not in selected][:max(0, size - len(selected))]
        if len(selected) != size:
            raise ValueError(f'Insufficient cross-family history for {target}')
        return selected
    selected = []
    for item in priority:
        if item not in answers:
            continue
        if {item, target} == {'v47', 'v48'} or any({item, previous} == {'v47', 'v48'} for previous in selected):
            continue
        selected.append(item)
        if size != 'all' and len(selected) == size:
            break
    return selected


def render_prompt(dataset, profile, history, target, instrument, native=False):
    survey = 'Pew Research Center American Trends Panel' if dataset == 'pew' else 'International Social Survey Programme'
    labels = NATIVE_LABELS if native else FIELD_LABELS[dataset]
    personal = [] if native else ['Country: United States']
    personal.extend(f'{labels[field]}: {value}' for field, value in profile.items() if value is not None)
    blocks = []
    for position, answer in enumerate(history, 1):
        item = instrument[answer['target']]
        blocks.append('\n'.join([f'#### Previous answer {position}', '', item['question'], '',
                                  f"Response options: {item['formatted_options']}", f"Selected option code: {answer['answer']}"]))
    item = instrument[target]
    prompt = '\n'.join(['### Instruction', '',
        f'You are participating in the {survey}. The personal information and previous survey answers below describe you as a survey respondent. Answer the final survey question as this respondent.',
        '', '### Personal Information', '', '\n'.join(personal), '', '### Previous Survey Answers', '',
        '\n\n'.join(blocks), '', '### Question', '', item['question'], '', '### Response Options', '',
        item['formatted_options'], '', '### Response', '', 'Return exactly one JSON object and no other text:',
        '{"option": "<selected option>"}'])
    if not history:
        prompt = prompt.replace(' The personal information and previous survey answers below describe you as a survey respondent.', '')
        prefix, after = prompt.split('### Previous Survey Answers\n\n', 1)
        prompt = prefix + '### Question\n\n' + after.split('### Question\n\n', 1)[1]
    return prompt


def build_jobs(rows, instrument, conditions, history_size=8, full_panel=False, assignments=None, shuffle_seed=20261017):
    dataset = rows[0]['dataset']
    if any(row['dataset'] != dataset for row in rows):
        raise ValueError('Build one survey at a time')
    evaluation = [row for row in rows if row.get('evaluation_cohort', True)]
    donors = donor_mapping(evaluation, dataset, shuffle_seed)
    training = [row for row in rows if not row.get('evaluation_cohort', True)]
    if training:
        donors.update(donor_mapping(training, dataset, shuffle_seed))
    by_identifier = {row['respondent_id']: row for row in rows}
    assignments = assignments or {}
    if dataset == 'issp' and any(condition not in ('History', 'Own', 'Shuffled') for condition in conditions):
        raise ValueError('ISSP conditions are History, Own and Shuffled')
    for row in rows if full_panel else evaluation:
        for target, spec in instrument.items():
            if target not in row['answers']:
                continue
            selected = select_history(instrument, row['answers'], target, dataset, history_size)
            history = [{'target': item, 'answer': row['answers'][item]} for item in selected]
            for condition in conditions:
                donor = by_identifier[donors[row['respondent_id']]] if condition.startswith('Shuffled') else row
                fields = DATASET_FIELDS[dataset]
                native = condition == 'Nemotron_all'
                if condition == 'History':
                    fields = ()
                elif condition.endswith('_10'):
                    fields = PEW_TEN
                elif condition == 'Own_7':
                    fields = PEW_SEVEN
                elif condition.endswith('_4'):
                    fields = PEW_FOUR if dataset == 'pew' else ISSP_FOUR
                elif condition == 'Own_political':
                    fields = fields + PEW_POLITICAL_FIELDS
                displayed = {field: donor['profile'][field] for field in fields}
                displayed_codes = {field: donor['profile_codes'][field] for field in fields}
                if condition in ('ACS_10', 'Nemotron_4', 'Nemotron_all'):
                    assignment = assignments.get(row['respondent_id'], {})
                    if condition not in assignment:
                        raise ValueError(f'External profile assignment missing for {condition}')
                    displayed = assignment[condition]
                    fields = tuple(displayed)
                    displayed_codes = displayed.copy()
                prompt = render_prompt(dataset, displayed, history, target, instrument, native)
                job = {key: copy.deepcopy(value) for key, value in row.items() if key != 'answers'}
                job.update(target=target, condition=condition, history=history, history_length=len(history), requested_history_length=history_size, actual_option=row['answers'][target],
                    options=list(spec['options']), option_labels=spec['options'], side_codes=spec['side_codes'],
                    displayed_profile=displayed, displayed_profile_codes=displayed_codes,
                    displayed_analysis_profile=displayed_codes, donor_analysis_profile=donor.get('analysis_profile', donor['profile_codes']),
                    profile_fields=list(fields), messages=[{'role': 'user', 'content': prompt}],
                    scoring_texts={code: json.dumps({'option': code},
                        separators=(',', ':') if dataset == 'pew' else (', ', ': '))
                        for code in spec['options']},
                    donor_id=donor['respondent_id'] if condition.startswith('Shuffled') else None)
                yield job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--respondents', type=Path, required=True)
    parser.add_argument('--instrument', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=['History', 'Own', 'Shuffled'])
    parser.add_argument('--history-size', default='8', help='Requested observed-answer count or all')
    parser.add_argument('--shuffle-seed', type=int, default=20261017)
    parser.add_argument('--full-panel', action='store_true', help='Include training respondents for reference fitting')
    parser.add_argument('--assignments', type=Path, help='Generated ACS or Nemotron assignments JSONL')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assignments = {row['respondent_id']: row['profiles'] for row in read_rows(args.assignments)} if args.assignments else None
    jobs = build_jobs(list(read_rows(args.respondents)), json.loads(args.instrument.read_text()), args.conditions,
                      'all' if args.history_size == 'all' else int(args.history_size), args.full_panel, assignments, args.shuffle_seed)
    write_rows(args.output, jobs)


if __name__ == '__main__':
    main()
