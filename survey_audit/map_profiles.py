"""Map and sample acquired ACS or Nemotron whole-person profiles."""

import argparse
import json
from pathlib import Path
import random
import zipfile

import numpy as np
import pandas as pd

from survey_audit import acs_mapping
from survey_audit.prepare_data import read_rows, write_rows
from survey_audit.survey_specifications import NEMOTRON_MAPPING, NEMOTRON_REVISION

PERSON_COLUMNS = ['SERIALNO', 'SPORDER', 'ST', 'REGION', 'ADJINC', 'PWGTP', 'AGEP',
                  'CIT', 'SEX', 'SCHL', 'RAC1P', 'HISP', 'NATIVITY', 'POBP', 'MAR',
                  'RELSHIPP', 'ESR', 'WKHP']
HOUSEHOLD_COLUMNS = ['SERIALNO', 'ST', 'REGION', 'ADJINC', 'TYPE', 'HINCP', 'FINCP', 'NP', 'HHT']
NATIVE_FIELDS = ('sex', 'age', 'marital_status', 'education_level', 'bachelors_field',
                 'occupation', 'city', 'state', 'zipcode')
VALUE_LABELS = {
    'marital_status': {'never_married': 'Never married', 'divorced': 'Divorced',
                      'widowed': 'Widowed', 'married_present': 'Married', 'separated': 'Separated'},
    'education_level': {'less_than_9th': 'Less than ninth grade',
        '9th_12th_no_diploma': 'Ninth through twelfth grade, no diploma',
        'high_school': 'High school graduate', 'some_college': 'Some college',
        'associates': 'Associate degree', 'bachelors': "Bachelor's degree", 'graduate': 'Graduate degree'},
    'bachelors_field': {'arts_humanities': 'Arts and humanities', 'stem': 'STEM',
        'business': 'Business', 'stem_related': 'STEM related', 'education': 'Education'},
}


def read_zip(path, columns):
    with zipfile.ZipFile(path) as archive:
        frames = [pd.read_csv(archive.open(name), usecols=columns,
                             dtype={'SERIALNO': str}, low_memory=False)
                  for name in sorted(archive.namelist()) if name.lower().endswith('.csv')]
    return pd.concat(frames, ignore_index=True)


def acs_population(persons_path, households_path):
    """Join households before filtering to adults in occupied U.S. households."""
    persons = read_zip(persons_path, PERSON_COLUMNS)
    households = read_zip(households_path, HOUSEHOLD_COLUMNS)
    partners = set(persons.loc[persons.RELSHIPP.isin([22, 24]), 'SERIALNO'])
    joined = persons.merge(households, on='SERIALNO', how='left', validate='many_to_one',
                           suffixes=('', '_HOUSEHOLD'), indicator=True)
    if not joined._merge.eq('both').all():
        raise ValueError('Missing household record')
    for field in ('ST', 'REGION', 'ADJINC'):
        if not joined[field].eq(joined[field + '_HOUSEHOLD']).all():
            raise ValueError(f'Person/household {field} differs')
    selected = joined.loc[joined.AGEP.ge(18) & joined.TYPE.eq(1)
                          & joined.ST.isin(acs_mapping.STATE_CODES)].copy()
    selected = selected.sort_values(['SERIALNO', 'SPORDER']).reset_index(drop=True)
    selected['HAS_UNMARRIED_PARTNER'] = selected.SERIALNO.isin(partners)
    population = acs_mapping.map_person_fields(selected)
    income = selected.FINCP.where(selected.FINCP.notna(), selected.HINCP)
    population['INCOME'] = pd.cut(income * selected.ADJINC / 1000000,
        [float('-inf'), 30000, 50000, 70000, 100000, float('inf')], right=False,
        labels=['Less than $30,000', '$30,000-$50,000', '$50,000-$75,000',
                '$75,000-$100,000', '$100,000 or more']).astype('string')
    weights = selected.PWGTP.to_numpy(dtype=np.int64)
    if len(population) != 2452515 or int(weights.sum()) != 247431811:
        raise ValueError('The 2019 national adult household population differs')
    return population, weights


def native_value(field, value):
    if value is None or (isinstance(value, str) and value.strip().lower() in
                         ('', 'null', 'none', 'n/a', 'not_applicable', 'not applicable')):
        return None
    if field == 'occupation' and value == 'no_occupation':
        return None
    if field in VALUE_LABELS:
        return VALUE_LABELS[field][value]
    if field == 'occupation':
        return str(value).replace('_', ' ').capitalize()
    return str(value)


def coarse_profile(record):
    regions = {state: region for region, states in NEMOTRON_MAPPING['CREGION'].items() for state in states}
    if record['country'] != 'USA' or record['age'] < 18 or record['state'] not in regions:
        return None
    age = next(group['value'] for group in NEMOTRON_MAPPING['AGE']
               if group['minimum_inclusive'] <= record['age'] and
               (group['maximum_exclusive'] is None or record['age'] < group['maximum_exclusive']))
    return dict(CREGION=regions[record['state']], SEX=NEMOTRON_MAPPING['SEX'][record['sex']],
                AGE=age, EDUCATION=NEMOTRON_MAPPING['EDUCATION'][record['education_level']])


def sample_acs(population, weights, recipients, seed):
    cumulative = np.cumsum(weights, dtype=np.int64)
    generator = np.random.Generator(np.random.PCG64(seed))
    tickets = generator.integers(0, int(weights.sum()), size=len(recipients), dtype=np.int64)
    positions = np.searchsorted(cumulative, tickets, side='right')
    for recipient, position in zip(recipients, positions):
        profile = {field: str(population.iloc[position][field]) for field in acs_mapping.FIELD_ORDER}
        yield dict(respondent_id=recipient['respondent_id'], profiles={'ACS_10': profile}, seed=seed)


def sample_nemotron(directory, recipients, seed):
    import pyarrow.parquet as parquet
    files = sorted(Path(directory).glob('*.parquet'))
    if len(files) != 11:
        raise ValueError('Provide all eleven Parquet shards of the pinned release')
    columns = ['uuid', 'country', *NATIVE_FIELDS]
    eligible = []
    for path in files:
        for record in parquet.read_table(path, columns=columns).to_pylist():
            coarse = coarse_profile(record)
            if coarse is not None:
                eligible.append((record, coarse))
    generator = random.Random(seed)
    for recipient in recipients:
        record, coarse = eligible[generator.randrange(len(eligible))]
        native = {field: native_value(field, record[field]) for field in NATIVE_FIELDS}
        native = {field: value for field, value in native.items() if value is not None}
        yield dict(respondent_id=recipient['respondent_id'], profiles={'Nemotron_4': coarse, 'Nemotron_all': native},
                   seed=seed, source_revision=NEMOTRON_REVISION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('acs', 'nemotron'), required=True)
    parser.add_argument('--respondents', type=Path, required=True)
    parser.add_argument('--persons', type=Path)
    parser.add_argument('--households', type=Path)
    parser.add_argument('--parquet-dir', type=Path)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--full-panel', action='store_true', help='Continue draws for reference training recipients')
    parser.add_argument('--population-output', type=Path, help='Optional complete mapped ACS population for identifiability')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = sorted(read_rows(args.respondents), key=lambda row: row['full_panel_profile_order'])
    recipients = [row for row in rows if row.get('evaluation_cohort', True)]
    if args.full_panel:
        recipients += [row for row in rows if not row.get('evaluation_cohort', True)]
    if args.source == 'acs':
        if not args.persons or not args.households:
            parser.error('ACS requires --persons and --households ZIP files')
        population, weights = acs_population(args.persons, args.households)
        write_rows(args.output, sample_acs(population, weights, recipients, args.seed))
        if args.population_output:
            write_rows(args.population_output, (dict(profile=dict(zip(population.columns, values)), weight=int(weight))
                for values, weight in zip(population.itertuples(index=False, name=None), weights)))
    else:
        if not args.parquet_dir:
            parser.error('Nemotron requires --parquet-dir')
        write_rows(args.output, sample_nemotron(args.parquet_dir, recipients, args.seed))


if __name__ == '__main__':
    main()
