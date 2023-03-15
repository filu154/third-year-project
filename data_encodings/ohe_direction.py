import numpy as np


def one_hot_encoding(x):
    nt_to_encode = {'A': [1, 0, 0, 0],
                    'T': [0, 1, 0, 0],
                    'C': [0, 0, 1, 0],
                    'G': [0, 0, 0, 1]}
    res = []
    for c in x:
        res.append(nt_to_encode.get(c, [0, 0, 0, 0]))
    return res


def represent_mismatches(target_sequence, grna_target_sequence):
    return [add_direction_mismatch(dna_nt, grna_nt) for dna_nt, grna_nt in zip(target_sequence, grna_target_sequence)]


def add_direction_mismatch(dna_nt, grna_nt):
    res = np.bitwise_or(dna_nt, grna_nt)
    for i in range(0, 4):
        if dna_nt[i] != grna_nt[i]:
            direction = [1, 0] if dna_nt[i] != 0 else [0, 1]
            res = np.append(res, direction)
            return res

    return np.append(res, [0, 0])


def encode(df):
    target_sequences, grna_target_sequences = df['target_sequence'], df['grna_target_sequence']

    ohe_target_sequences = map(lambda x: one_hot_encoding(x), target_sequences)
    ohe_grna_target_sequences = map(lambda x: one_hot_encoding(x), grna_target_sequences)

    return [represent_mismatches(target_sequence, grna_target_sequence) for target_sequence, grna_target_sequence in zip(ohe_target_sequences, ohe_grna_target_sequences)]
