import pandas as pd
import numpy as np
import scipy.stats
import sklearn
import tensorflow
from tensorflow import keras
from keras import datasets, layers, models

pd.set_option('display.max_rows', None)
df = pd.read_csv('../data/100720.csv', index_col='id')
df1 = df[['target_sequence', 'grna_target_sequence', 'cleavage_freq']].copy()

# Delete entries that do not have 23 nt
df1 = df1[df1['target_sequence'].apply(lambda x: len(x)==23)]
df1 = df1[df1['grna_target_sequence'].apply(lambda x: len(x)==23)]
df1 = df1[df1['cleavage_freq'].apply(lambda x: not pd.isnull(x))]

nt_to_encode = {'A': [1, 0, 0, 0],
                'T': [0, 1, 0, 0],
                'C': [0, 0, 1, 0],
                'G': [0, 0, 0, 1]}


def one_hot_encoding(x):
    res = []
    for c in x:
        res.append(nt_to_encode.get(c, [0, 0, 0, 0]))
    return res


df1['target_sequence'] = df1['target_sequence'].apply(one_hot_encoding)
df1['grna_target_sequence'] = df1['grna_target_sequence'].apply(one_hot_encoding)
df1['tmp'] = df1.index

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


df1 = sklearn.utils.shuffle(df1)
df1.reset_index(inplace=True,drop=True)


x_data = [represent_mismatches(target_sequence, grna_target_sequence) for target_sequence, grna_target_sequence in zip(df1['target_sequence'], df1['grna_target_sequence'])]
y_data = df1['cleavage_freq']


N = len(df1.index)
N_train = int(0.8 * N)
X_train, y_train, X_test, y_test = x_data[:N_train], y_data[:N_train], x_data[N_train:], y_data[N_train:]
print(len(X_test), len(y_test))

kernel_init = keras.initializers.RandomNormal(mean=0.0,stddev=0.05)
bias_init = keras.initializers.Constant(0.1)
input_shape = (23, 6, 1)

model = models.Sequential()

model.add(layers.Conv2D(filters=40, kernel_size=(4, 6), padding='valid', activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init,
                        input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,1)))
model.add(layers.Flatten())
model.add(layers.Dense(units=125, activation='relu',
                       kernel_initializer=kernel_init,
                       bias_initializer=bias_init))
model.add(layers.Dense(units=23, activation='relu',
                       kernel_initializer=kernel_init,
                       bias_initializer=bias_init))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=1, activation='sigmoid'))


# model.summary()

X_train = keras.backend.expand_dims(X_train)
X_test = keras.backend.expand_dims(X_test)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=keras.losses.BinaryCrossentropy(), metrics='mae')
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

predicts = model.predict(X_test)
print(scipy.stats.spearmanr(y_test, predicts))

# model.evaluate(X_test, y_test)

# model.save('Saved Models/model1')

# test_model = keras.models.load_model('Saved Models/model1')
# input_test = represent_mismatches(one_hot_encoding('CACCCATAAAGATGAGACGCTGG'), one_hot_encoding('GACGCATAAAGATGAGACGCTGG'))
# input_test = keras.backend.expand_dims(input_test)
# print(input_test.shape)
# print(test_model.predict(np.expand_dims(input_test, axis=0)))
