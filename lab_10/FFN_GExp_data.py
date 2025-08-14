# FFN

# Libraries
import pandas as pd
import torch as nn
from sklearn.model_selection import train_test_split


# Import data
df = pd.read_table('GD462.GeneQuantRPKM.50FN.samplename.resk10.txt', sep='\t')
lm_id = pd.read_csv('map_lm.txt', sep='\t', header=None)
tg_id = pd.read_csv('map_tg.txt', sep='\t', header=None)

tg_id.columns = lm_id.columns = ['gene_name', 'Gene_Symbol', 'loc']


## Data preprocessing
print(
    df.describe(),'\n'
)

print(
    df.head(),'\n',
    df.columns,'\n',
    df.shape
)

## Check for NA rows

print('---'*9)
print("\nChecking NA rows...")
na_rows = df.isna().any(axis=1)
num_na_rows = df[na_rows]


if num_na_rows.shape[0] == 0:
    print("\nNo, NA rows found")
else:
    print(f'Found {num_na_rows.shape[0]} NA vals')

print(lm_id)

# ## Subset data for Landmark and target genes
df['Gene_Symbol'] = df['Gene_Symbol'].str.split(".", expand=True)
print(df['Gene_Symbol'].head(5))

quit()
lm_df = df[:,df['Gene_Symbol']==lm_id["Gene_Symbol"]]
print(lm_df)
quit()
# train test split
# train_test_split()

# Train - val split
# Model training
# Hyper parameter tuning
# Generate stats
# Train full model
# Generate statistics
# Conclusion