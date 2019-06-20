
from dataset import data_loader_k200_serial as data_loader
import numpy as np
import json
import train, infer

# to reproduce same training result
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

if __name__ == "__main__":

    train_start_date = '2018-01-01'
    train_end_date = '2018-06-30'
    test_start_date = '2018-07-01'
    test_end_date = '2018-08-31'
    x_window_length = 10
    # code = '009150'  # 삼성전기

    featured_columns = ['Marcap', 'Amount', 'ForeignRatio']
    lognorm_columns = [True, True, False]
    (X1_train, X2_train, X3_train, y_train), (X1_test, X2_test, X3_test, y_test) = data_loader.load_data(
        train_start_date, train_end_date, test_start_date, test_end_date, x_window_length, featured_columns,
        lognorm_columns)
    #X1_val, X2_val, X3_val, y_val = X1_test, X2_test, X3_test, y_test

    print(X1_train.shape)
    print(X2_train.shape)
    print(X3_train.shape)
    print(y_train.shape)
    print(X1_test.shape)
    print(X2_test.shape)
    print(X3_test.shape)
    print(y_test.shape)

    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # X_val = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #
    # model = train.train_model(X_train, y_train, X_val, y_val)
    #
    # y_hats = infer.infer_testset(model, X_test, y_test)
    # infer.draw_result_plot(y_hats, y_test)