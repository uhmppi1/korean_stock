from keras.models import Sequential
from keras.layers import Dense, LSTM

def get_model():
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
    model.add(Dense(1))
    model.summary()


    return model
