"""Prepare acquired survey sources and questionnaire metadata for local analysis."""

import argparse
import ast
from collections import defaultdict
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import zipfile

import pandas as pd

from survey_audit.survey_specifications import (
    TARGETS, PEW_FIELDS, PEW_POLITICAL_FIELDS, PEW_ELIGIBILITY_FIELDS,
    ISSP_FIELDS, ISSP_CORE_FIELDS, OCCUPATION_LABELS,
)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def clean_label(label):
    prefix, separator, remainder = str(label).partition('. ')
    if separator and prefix.lstrip('-').isdigit():
        label = remainder
    return str(label).strip().rstrip(' ,;:.')


def normalize_scalar(value):
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def assign_folds(rows, dataset, supplied=None):
    """Restore saved folds from a full JSON mapping or Pew analysis CSV."""
    if supplied and Path(supplied).suffix.lower() == '.csv':
        if dataset != 'pew':
            raise ValueError('QKEY fold CSV files apply to Pew only')
        with Path(supplied).open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['QKEY', 'fold']:
                raise ValueError('Pew fold CSV requires exactly QKEY,fold columns')
            saved = list(reader)
        mapping = {normalize_scalar(row['QKEY']): int(row['fold']) for row in saved}
        if len(mapping) != len(saved) or any(fold not in range(5) for fold in mapping.values()):
            raise ValueError('Pew fold CSV requires unique QKEYs and folds 0 through 4')
        expected = {str(row.get('source_respondent_id', row['respondent_id']))
                    for row in rows if row.get('evaluation_cohort', True)}
        if set(mapping) != expected:
            raise ValueError('Pew fold CSV must match the complete analysis cohort')
        assign_folds(rows, dataset)
        for row in rows:
            identifier = str(row.get('source_respondent_id', row['respondent_id']))
            if identifier in mapping:
                row['fold'] = mapping[identifier]
        return rows
    if supplied:
        mapping = json.loads(Path(supplied).read_text())
    elif dataset == 'pew':
        ordered = sorted((row['respondent_id'] for row in rows), key=lambda value: (
            hashlib.sha256(f'72042|{value}'.encode()).digest(), value))
        mapping = {value: position % 5 for position, value in enumerate(ordered)}
    else:
        grouped = defaultdict(list)
        counts = {}
        for row in rows:
            signature = tuple(row['profile'][field] for field in ISSP_CORE_FIELDS)
            grouped[signature].append(row['respondent_id'])
            counts[row['respondent_id']] = sum(target in row['answers'] for target in list(TARGETS['issp'])[:24])
        groups = [tuple(sorted(members)) for _, members in sorted(grouped.items())]
        random.Random(20260904).shuffle(groups)
        groups.sort(key=len, reverse=True)
        people, cells, group_counts = [0] * 5, [0] * 5, [0] * 5
        mapping = {}
        for members in groups:
            fold = min(range(5), key=lambda index: (people[index], cells[index], group_counts[index], index))
            mapping.update({person: fold for person in members})
            people[fold] += len(members)
            cells[fold] += sum(counts[person] for person in members)
            group_counts[fold] += 1
    for row in rows:
        row['fold'] = int(mapping[row['respondent_id']])
        if row['fold'] not in range(5):
            raise ValueError('Five-fold assignment requires values 0 through 4')
    return rows


def prepare_pew(responses_path, info_path):
    """Apply the twelve-field English-language eligibility and fixed sample."""
    instrument = copy.deepcopy(TARGETS['pew'])
    with Path(info_path).open(encoding='utf-8-sig', newline='') as stream:
        information = {row['key']: row for row in csv.DictReader(stream)}
    for target, spec in instrument.items():
        source = information[target]
        references = ast.literal_eval(source['references'])
        ordinal = spec['ordinal']
        options = {str(index + 1): str(label).strip() for index, label in enumerate(references[:len(ordinal)])}
        spec.update(question=source['question'], options=options,
                    formatted_options='; '.join(f'{code} {label}' for code, label in options.items()))
    with Path(responses_path).open(encoding='utf-8-sig', newline='') as stream:
        source_rows = sorted(csv.DictReader(stream), key=lambda row: normalize_scalar(row['QKEY']))
    rows = []
    missing = {'', 'nan', 'none', 'null', 'missing', 'no answer', 'refused', "don't know", 'dont know',
               'not asked', 'not applicable', 'not answered', 'not available', 'skipped', 'unknown'}
    for source in source_rows:
        if source['LANG_W54'].strip().casefold() != 'english':
            continue
        if any(source[field].strip().casefold() in missing for field in PEW_ELIGIBILITY_FIELDS):
            continue
        weight = float(source['WEIGHT_W54'])
        if not weight > 0:
            continue
        answers = {}
        for target, spec in instrument.items():
            value = normalize_scalar(source[target])
            for index, coordinate in enumerate(spec['ordinal']):
                code = str(index + 1)
                if value == normalize_scalar(coordinate) or value.casefold() == spec['options'][code].strip().casefold():
                    answers[target] = code
                    break
        profile = {field: source[field].strip() or 'Not recorded' for field in PEW_FIELDS + PEW_POLITICAL_FIELDS}
        rows.append(dict(respondent_id=normalize_scalar(source['QKEY']), dataset='pew', weight=weight,
                         profile=profile, profile_codes=profile.copy(), analysis_profile=profile.copy(),
                         answers=answers, full_panel_profile_order=len(rows) + 1))
    if len(rows) != 6088:
        raise ValueError(f'The frozen Wave 54 source requires 6088 eligible rows; found {len(rows)}')
    selected = set(random.Random(20261019).sample(range(1, len(rows) + 1), 1000))
    for row in rows:
        row['evaluation_cohort'] = row['full_panel_profile_order'] in selected
    return rows, instrument


def prepare_issp(archive_path, questionnaire_path):
    """Read the integrated Stata archive and the U.S. source questionnaire."""
    from survey_audit.questionnaire import read_questionnaire, QUESTION_SOURCES, BATTERY_ITEM_STARTS
    with zipfile.ZipFile(archive_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith('.dta'))
        payload = archive.read(member)
    with pd.io.stata.StataReader(io.BytesIO(payload)) as reader:
        label_sets = reader.value_labels()
        labels = {field: label_sets[field.upper()] for field in ISSP_FIELDS if field != 'AGE'}
        columns = ['CASEID', 'c_alphan', 'WEIGHT_COM', 'PARTIALS', *ISSP_FIELDS, *TARGETS['issp']]
        frame = reader.read(columns=columns, convert_categoricals=False, preserve_dtypes=False)
    frame = frame.loc[frame.c_alphan.eq('US') & frame.PARTIALS.eq(1) & frame.AGE.gt(0)
                      & frame[list(ISSP_CORE_FIELDS[1:])].ge(0).all(axis=1)].copy()
    frame['respondent_id'] = frame.CASEID.astype('int64').astype(str)
    frame = frame.sort_values('respondent_id')
    if len(frame) != 1513:
        raise ValueError(f'The frozen ISSP source requires 1513 eligible rows; found {len(frame)}')
    questionnaire = read_questionnaire(questionnaire_path)
    instrument = copy.deepcopy(TARGETS['issp'])
    for target, spec in instrument.items():
        variable, battery, start = QUESTION_SOURCES[target]
        source = questionnaire[variable]
        introduction = questionnaire[battery]['question'].split(BATTERY_ITEM_STARTS[battery], 1)[0].strip() if battery else ''
        text = source['question']
        if start:
            text = start + text.split(start, 1)[1]
        option_labels = [label for _, label in source['options'] if "Can't choose" not in label]
        codes = list(spec['options'])
        if len(option_labels) != len(codes):
            raise ValueError(f'Questionnaire option count changed: {target}')
        options = dict(zip(codes, option_labels))
        formatted = '; '.join(label if re.match(rf'^{code}(?:\s|$)', label) else f'{code} {label}' for code, label in options.items())
        spec.update(question='\n\n'.join(value for value in (introduction, text) if value),
                    options=options, formatted_options=formatted)
    rows = []
    for source in frame.to_dict('records'):
        profile, codes = {}, {}
        for field in ISSP_FIELDS:
            value = source[field]
            if pd.isna(value) or value < 0:
                profile[field] = codes[field] = None
                continue
            code = int(value)
            codes[field] = str(code // 1000 if field == 'ISCO08' else code)
            profile[field] = str(code) if field == 'AGE' else OCCUPATION_LABELS[code // 1000] if field == 'ISCO08' else clean_label(labels[field][code])
        answers = {target: str(int(source[target])) for target, spec in instrument.items()
                   if pd.notna(source[target]) and str(int(source[target])) in spec['options']}
        rows.append(dict(respondent_id=source['respondent_id'], dataset='issp', weight=float(source['WEIGHT_COM']),
                         profile=profile, profile_codes=codes, analysis_profile=codes.copy(), answers=answers,
                         full_panel_profile_order=len(rows) + 1, evaluation_cohort=True))
    return rows, instrument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('pew', 'issp'), required=True)
    parser.add_argument('--responses', type=Path, help='OpinionQA Wave 54 responses.csv')
    parser.add_argument('--info', type=Path, help='OpinionQA Wave 54 info.csv')
    parser.add_argument('--archive', type=Path, help='ISSP ZA10010 v1.0.0 Stata ZIP')
    parser.add_argument('--questionnaire', type=Path, help='ISSP U.S. English questionnaire PDF')
    parser.add_argument('--folds', type=Path, help='Full ID-to-fold JSON mapping or Pew analysis QKEY,fold CSV')
    parser.add_argument('--identifiers', type=Path, help='Optional source ID to original analysis ID mapping')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instrument-out', type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == 'pew':
        if args.responses is None or args.info is None:
            parser.error('Pew requires --responses and --info')
        rows, instrument = prepare_pew(args.responses, args.info)
    else:
        if args.archive is None or args.questionnaire is None:
            parser.error('ISSP requires --archive and --questionnaire')
        rows, instrument = prepare_issp(args.archive, args.questionnaire)
    for row in rows:
        row['source_respondent_id'] = row['respondent_id']
    if args.identifiers:
        identifiers = json.loads(args.identifiers.read_text())
        for row in rows:
            row['respondent_id'] = identifiers[row['respondent_id']]
    assign_folds(rows, args.dataset, args.folds)
    write_rows(args.output, rows)
    args.instrument_out.parent.mkdir(parents=True, exist_ok=True)
    args.instrument_out.write_text(json.dumps(instrument, ensure_ascii=False, indent=2) + '\n')
    print(f'Prepared {len(rows)} eligible respondents and {len(instrument)} targets')


if __name__ == '__main__':
    main()
