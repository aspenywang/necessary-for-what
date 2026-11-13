"""Map 2019 adult household person records to the ten Pew profile fields."""

import pandas
FIELD_ORDER = ('CREGION', 'SEX', 'AGE', 'EDUCATION', 'RACE', 'CITIZEN', 'F_NATIVITY', 'MARITAL', 'F_E3', 'INCOME')
DICTIONARY_PAGES = {'CREGION': '31', 'SEX': '43', 'AGE': '32', 'EDUCATION': '42-43', 'RACE': '61-62, 103', 'CITIZEN': '32', 'F_NATIVITY': '86, 97-101', 'MARITAL': '37, 41-42', 'F_E3': '43-44, 55', 'INCOME': '2, 14'}
STATE_CODES = frozenset({1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56})
OTHER_TERRITORY_CODES = frozenset({60, 66, 69, 78})
FOREIGN_BIRTHPLACE_CODES = frozenset({100, 102, 103, 104, 105, 106, 108, 109, 110, 116, 117, 118, 119, 120, 126, 127, 128, 129, 130, 132, 134, 136, 137, 138, 139, 140, 142, 147, 148, 149, 150, 151, 152, 154, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 200, 202, 203, 205, 206, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 222, 223, 224, 226, 228, 229, 231, 233, 235, 236, 238, 239, 240, 242, 243, 245, 246, 247, 248, 249, 253, 254, 300, 301, 303, 310, 311, 312, 313, 314, 315, 316, 321, 323, 324, 327, 328, 329, 330, 332, 333, 338, 339, 340, 341, 343, 344, 360, 361, 362, 363, 364, 365, 368, 369, 370, 372, 373, 374, 399, 400, 407, 408, 412, 414, 416, 417, 420, 421, 423, 425, 427, 429, 430, 436, 440, 442, 444, 447, 448, 449, 451, 453, 454, 456, 457, 459, 460, 461, 462, 463, 464, 467, 468, 469, 501, 508, 511, 512, 515, 523, 527})

def _check_codes(frame, column, allowed, allow_missing=False):
    """Reject source codes outside the approved 2019 adult universe."""
    invalid = ~frame[column].isin(allowed)
    if allow_missing:
        invalid &= frame[column].notna()
    if invalid.any():
        values = frame.loc[invalid, column].unique().tolist()
        raise ValueError(f'Unexpected {column} values: {values}')

def map_person_fields(frame):
    """Return ten string columns, preserving the supplied frame's index.

    Inputs are numeric PUMS columns plus boolean HAS_UNMARRIED_PARTNER,
    computed across all household members before the adult-universe filter.
    Missing WKHP follows the approved 'otherwise Not employed' rule. All
    other required source values must be present.
    NATIVITY validates source coding and does not override birthplace.
    """
    code_sets = {'REGION': range(1, 5), 'SEX': (1, 2), 'AGEP': range(18, 100), 'SCHL': range(1, 25), 'RAC1P': range(1, 10), 'HISP': range(1, 25), 'CIT': range(1, 6), 'NATIVITY': (1, 2), 'MAR': range(1, 6), 'RELSHIPP': range(20, 37), 'ESR': range(1, 7), 'POBP': STATE_CODES | OTHER_TERRITORY_CODES | FOREIGN_BIRTHPLACE_CODES | {72, 554}, 'ADJINC': (1010145,), 'HAS_UNMARRIED_PARTNER': (True, False)}
    for (column, allowed) in code_sets.items():
        _check_codes(frame, column, allowed)
    _check_codes(frame, 'WKHP', range(1, 100), allow_missing=True)
    if not frame['HINCP'].between(-59999, 9999999).fillna(False).all():
        raise ValueError('HINCP must be present and within the 2019 range')
    result = pandas.DataFrame(index=frame.index)
    result['CREGION'] = frame['REGION'].map({1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'})
    result['SEX'] = frame['SEX'].map({1: 'Male', 2: 'Female'})
    result['AGE'] = pandas.cut(frame['AGEP'], [18, 30, 50, 65, 100], right=False, labels=['18-29', '30-49', '50-64', '65+'])
    result['EDUCATION'] = pandas.cut(frame['SCHL'], [1, 16, 18, 20, 21, 22, 25], right=False, labels=['Less than high school', 'High school graduate', 'Some college, no degree', "Associate's degree", 'College graduate/some postgrad', 'Postgraduate'])
    result['RACE'] = 'Other'
    result.loc[frame['RAC1P'].eq(9), 'RACE'] = 'Mixed Race'
    result.loc[frame['HISP'].between(2, 24), 'RACE'] = 'Hispanic'
    for (code, label) in {1: 'White', 2: 'Black', 6: 'Asian'}.items():
        result.loc[frame['RAC1P'].eq(code), 'RACE'] = label
    result['CITIZEN'] = frame['CIT'].map({1: 'Yes', 2: 'Yes', 3: 'Yes', 4: 'Yes', 5: 'No'})
    result['F_NATIVITY'] = 'Another country'
    result.loc[frame['POBP'].isin(STATE_CODES), 'F_NATIVITY'] = 'U.S.'
    result.loc[frame['POBP'].eq(72), 'F_NATIVITY'] = 'Puerto Rico'
    result.loc[frame['POBP'].isin(OTHER_TERRITORY_CODES), 'F_NATIVITY'] = 'Other U.S. territory'
    result.loc[frame['POBP'].eq(554) & frame['CIT'].eq(2), 'F_NATIVITY'] = 'Other U.S. territory'
    result['MARITAL'] = frame['MAR'].map({1: 'Married', 2: 'Widowed', 3: 'Divorced', 4: 'Separated', 5: 'Never been married'})
    partner = frame['RELSHIPP'].isin((22, 24)) | frame['RELSHIPP'].eq(20) & frame['HAS_UNMARRIED_PARTNER']
    result.loc[partner, 'MARITAL'] = 'Living with a partner'
    employed = frame['ESR'].isin((1, 2, 4, 5))
    result['F_E3'] = 'Not employed'
    result.loc[employed & frame['WKHP'].ge(35), 'F_E3'] = 'Full-time'
    result.loc[employed & frame['WKHP'].lt(35), 'F_E3'] = 'Part-time'
    income = frame['HINCP'] * frame['ADJINC'] / 1000000
    result['INCOME'] = pandas.cut(income, [float('-inf'), 30000, 50000, 70000, 100000, float('inf')], right=False, labels=['Less than $30,000', '$30,000-$50,000', '$50,000-$75,000', '$75,000-$100,000', '$100,000 or more'])
    return result.loc[:, FIELD_ORDER].astype('string')
