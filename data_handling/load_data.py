import pandas as pd
import sklearn
from data_encodings import ohe_direction
from tensorflow import keras

def load_data(data_path):
    df = pd.read_csv(data_path, index_col='id')

    # Shuffle data
    df = sklearn.utils.shuffle(df)
    df.reset_index(inplace=True, drop=True)

    # get data and labels
    x_data = ohe_direction.encode(df)
    y_data = df['cleavage_freq']

    return x_data, y_data


def split_train_test(x_data, y_data):
    # split data into
    N = len(y_data)
    N_train = int(0.8 * N)
    X_train, y_train, X_test, y_test = x_data[:N_train], y_data[:N_train], x_data[N_train:], y_data[N_train:]

    X_train = keras.backend.expand_dims(X_train)
    X_test = keras.backend.expand_dims(X_test)

    return (X_train, y_train, X_test, y_test)