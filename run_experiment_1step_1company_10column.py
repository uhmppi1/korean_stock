from dataset import data_loader_k200_serial_scaled as data_loader
from model import simple_lstm
import numpy as np
import json
import train, infer

# to reproduce same training result
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

if __name__ == "__main__":

    train_start_date = '2015-01-01'
    # train_start_date = '2018-01-01'
    train_end_date = '2018-06-30'
    test_start_date = '2018-07-01'
    test_end_date = '2018-08-31'
    x_window_length = 10
    company_code_list = ['009150']   # code = '009150'  # 삼성전기

    # dataset_pickle_path = 'dataset/pickle/dataset_1step_1company_10col.pkl'
    dataset_pickle_path = 'dataset/pickle/dataset_1step_1company_10col_3year.pkl'

    featured_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount',
                          'Marcap', 'MarcapRatio', 'ForeignShares', 'ForeignRatio']
    lognorm_columns = [True, True, True, True, True, True,
                       True, False, True, False]

    dataloader = data_loader.DataLoader(dataset_pickle_path)

    (X1_train, X2_train, X3_train, y_train), (X1_test, X2_test, X3_test, y_test) = dataloader.load_data(
        train_start_date, train_end_date, test_start_date, test_end_date, x_window_length, featured_columns,
        lognorm_columns, company_code_list)
    #X1_val, X2_val, X3_val, y_val = X1_test, X2_test, X3_test, y_test

    print(X1_train.shape)
    print(X2_train.shape)
    print(X3_train.shape)
    print(y_train.shape)
    print(X1_test.shape)
    print(X2_test.shape)
    print(X3_test.shape)
    print(y_test.shape)

    X_train = X1_train
    y_train = y_train
    X_test = X1_test
    y_test = y_test
    X_val = X_test
    y_val = y_test

    model = simple_lstm.get_model(50, (10, 10))
    model = train.train_model(model, X_train, y_train, X_val, y_val)

    y_hats = infer.infer_testset(model, X_test, y_test, (10, 10))

    y_test_invscaled = dataloader.get_invertscaled_values('Marcap', y_test)
    y_hats_invscaled = dataloader.get_invertscaled_values('Marcap', y_hats)

    infer.draw_result_plot(y_hats_invscaled, y_test_invscaled)