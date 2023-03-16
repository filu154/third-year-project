from sklearn.model_selection import KFold

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