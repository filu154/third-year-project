import pandas as pd
import numpy as np
import sklearn
import scipy
from tensorflow import keras
from data_encodings import ohe_direction
from sklearn.model_selection import KFold

# Make sure to change here the model you want to train
from models.model1 import Model1

# Constants to change depending on what encoding and model you want to train
# Not all encodings and models are compatible

data_path = 'data/fixed_length_23.csv'
optimizer = keras.optimizers.Adam(learning_rate=1e-4)
loss = keras.losses.BinaryCrossentropy()
metrics = "mae"
batch_size=64
epochs=20
validation_split=0.1

# Load data
df = pd.read_csv(data_path, index_col='id')

# Shuffle data
df = sklearn.utils.shuffle(df)
df.reset_index(inplace=True, drop=True)

# Prepare data for training and testing
x_data = ohe_direction.encode(df)
y_data = df['cleavage_freq']

# N = len(df.index)
# N_train = int(0.8 * N)
# X_train, y_train, X_test, y_test = x_data[:N_train], y_data[:N_train], x_data[N_train:], y_data[N_train:]
#
# X_train = keras.backend.expand_dims(X_train)
# X_test = keras.backend.expand_dims(X_test)
#
# # Load model you want and train
# model = Model1()
# model.build()
# model.compile(optimizer, loss, metrics)
# model.fit(X_train, y_train, batch_size, epochs, validation_split)

# # Get metrics
# predicts = model.predict(X_test)
# print(scipy.stats.spearmanr(y_test, predicts))

kf = KFold(n_splits=5)
for train, test in kf.split(x_data):
    print(train)
    training_data = keras.backend.expand_dims(np.array(x_data)[train])
    training_labels = y_data[train]

    testing_data = keras.backend.expand_dims(np.array(x_data)[test])
    testing_labels = y_data[test]

    model = None
    model = Model1()
    model.build()
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    loss = keras.losses.BinaryCrossentropy()
    model.compile(optimizer, loss, metrics)
    model.fit(training_data, training_labels, batch_size, epochs, validation_split)
    predicts = model.predict(testing_data)
    print(scipy.stats.spearmanr(testing_labels, predicts))

