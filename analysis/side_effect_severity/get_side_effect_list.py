import pandas as pd

# Load data
polypharm_df = pd.read_csv('../../data/raw/bio-decagon-combo.csv')

# Grab side effects
se_col = 'Side Effect Name'
side_effect_df = pd.DataFrame(
    polypharm_df[se_col].unique(), 
    columns=[se_col]
)

# Stitch corresponding codes on
side_effect_df = side_effect_df.merge(
    polypharm_df[[se_col, 'Polypharmacy Side Effect']], 
    on=se_col
).drop_duplicates()

# Save
side_effect_df.columns = ['Side Effect', 'Code']
side_effect_df.to_csv('side_effects.csv', index=False)