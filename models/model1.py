import tensorflow
from tensorflow import keras
from data_encodings import ohe_direction
from keras import datasets, layers, models


class Model1():
    _kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    _bias_init = keras.initializers.Constant(0.1)
    _input_shape = (23, 6, 1)
    _kernel_shapes = [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]

    def __init__(self):
        self.model = models.Sequential()

    def build(self):
        input_layer = layers.Input(self._input_shape)

        conv_layers = []
        for ks in self._kernel_shapes:
            layer = layers.Conv2D(filters=10, kernel_size=ks, padding='valid', activation='relu',
                                  kernel_initializer=self._kernel_init,
                                  bias_initializer=self._bias_init,
                                  input_shape=self._input_shape)
            conv_layers.append(layer(input_layer))

        conc_layer = layers.Concatenate(axis=1)(conv_layers)
        batch_normalization = layers.BatchNormalization()(conc_layer)
        max_pooling_2d = layers.MaxPooling2D(pool_size=(2, 1))(batch_normalization)
        flatten = layers.Flatten()(max_pooling_2d)
        dense_1 = layers.Dense(units=200, activation='relu',
                               kernel_initializer=self._kernel_init,
                               bias_initializer=self._bias_init)(flatten)
        dense_2 = layers.Dense(units=23, activation='relu',
                               kernel_initializer=self._kernel_init,
                               bias_initializer=self._bias_init)(dense_1)
        dropout = layers.Dropout(0.1)(dense_2)
        output = layers.Dense(units=1, activation='sigmoid')(dropout)

        self.model = models.Model(inputs=input_layer, outputs=output)

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        self.model.summary()

    def fit(self, X_train, y_train, batch_size, epochs, validation_split):
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def predict(self, X_test):
        return self.model.predict(X_test)
