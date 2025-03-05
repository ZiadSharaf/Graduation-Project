NJOBS = 5

# ordinal features
ORDS = ['smoking', 'drinking']
# nominal features
NOMS = ['sex', 'dm', 'exercise']
# categorical features
CATS = NOMS + ORDS

# discrete features
DISCS = ['age', 'ast', 'ggt']
# continuous features
CONTS = ['height', 'fm', 'wc', 'hdl', 'tc', 'tg', 'fpg', 'hba1c', 'sbp']
# numercial features
NUMS = DISCS + CONTS

CAT_DICT = {
    'drinking': ('None/Small', 'Light', 'Moderate'),
    'smoking': ("None", "Past", "Present"),
    'sex': ("Female", "Male"),
    'exercise': ("Plays exercies less often once a week", "Plays exercise at least once a week"),
    'dm': ("Doesn't have diabetes", "Has diabetes"),
    'nafld': ("Doesn't have NAFLD", "Has NAFLD")
}

TEXT_DICT = {
    'drinking': 'Drinking status',
    'smoking': 'Smoking status',
    'sex': 'Sex',
    'age': 'Age',
    'ast': 'AST (U/L)',
    'ggt': 'GGT (U/L)',
    'height': 'Height (cm)',
    'fm': 'Fat Mass (kg)',
    'wc': 'Waist Circumference (cm)',
    'hdl': 'HDL (mmol/L)',
    'tc': 'TC (mmol/L)',
    'tg': 'TG (mmol/L)',
    'fpg': 'FPG (mmol/L)',
    'hba1c': 'HbA1c (%)',
    'sbp': 'SBP (mmHg)'
}
