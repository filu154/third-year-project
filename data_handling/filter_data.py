import pandas as pd

def filter_fixed_length(data_path, new_file_path):
    df = pd.read_csv(data_path, index_col='id')

    # Delete entries that do not have 23 nt
    df = df[df['target_sequence'].apply(lambda x: len(x)==23)]
    df = df[df['grna_target_sequence'].apply(lambda x: len(x)==23)]
    df = df[df['cleavage_freq'].apply(lambda x: not pd.isnull(x))]

    df.to_csv(path_or_buf=new_file_path, index=False)


filter_fixed_length('../data/100720.csv', '../data/fixed_length_23.csv')