import scipy
from tensorflow import keras
from data_handling import load_data

# Make sure to change here the model you want to train
from models.model1 import Model1

# Add here encodings that you want to use
from data_encodings import ohe_direction

# Constants to change depending on what encoding and model you want to train
# Not all encodings and models are compatible

data_path = '../data/fixed_length_23.csv'
optimizer = keras.optimizers.experimental.Adam(learning_rate=1e-4)
loss = keras.losses.BinaryCrossentropy()
metrics = "mae"
batch_size=16
epochs=20
validation_split=0.1


# Load whole dataset
x_data_dict, y_data = load_data.load_data(data_path)

# Encode data
x_data = ohe_direction.encode(x_data_dict)

# Split data into training and testing dataset
X_train, y_train, X_test, y_test = load_data.split_train_test(x_data, y_data)

# Load model you want and train
model = Model1()
model.build()
model.compile(optimizer, loss, metrics)
model.fit(X_train, y_train, batch_size, epochs, validation_split)

# Get metrics
model.get_metrics(X_train, y_train)


