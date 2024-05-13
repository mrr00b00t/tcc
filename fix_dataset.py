import pandas as pd

df = pd.read_csv('datasets/pima/pima2.tsv.gz', compression='gzip', header=0, sep="\t")

columns_to_fix = ['plasma glucose', 'Diastolic blood pressure', 'Triceps skin fold thickness', 'Body mass index']

for column in columns_to_fix:
    m = df.loc[(df['target'] == 0) & (df[column] != 0)][column].values.mean()
    df.loc[(df['target'] == 0) & (df[column] == 0), column] = m

    m = df.loc[(df['target'] == 1) & (df[column] != 0)][column].values.mean()
    df.loc[(df['target'] == 1) & (df[column] == 0), column] = m

df.to_csv('datasets/pima/pima.tsv.gz', sep='\t', compression='gzip', index=False)