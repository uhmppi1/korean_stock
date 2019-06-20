from keras.models import Sequential
from keras.layers import Dense, LSTM

def get_model(hidden_size=50, input_shape=(10, 1)):
    # define model
    model = Sequential()
    model.add(LSTM(hidden_size, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.summary()

    return model
