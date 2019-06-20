import json


def train_model(model, X_train, y_train, X_val, y_val, epochs=200):
    # model = simple_lstm.get_model()
    # model.summary()
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_val, y_val))

    model_json = model.to_json()
    with open("checkpoint/simple_model.json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights("checkpoint/simple_model.h5")

    return model


def train_seq2seq_model(model, input_data_1, input_data_2, teacher_data, target_data, batch_size=32, epochs=200):
    # model = simple_lstm.get_model()
    # model.summary()
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit([input_data_1, input_data_2, teacher_data], target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)

    model_json = model.to_json()
    with open("checkpoint/seq2seq_model.json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights("checkpoint/seq2seq_model.h5")

    return model