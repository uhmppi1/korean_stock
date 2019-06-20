import dataset.marcap_utils as mu
import numpy as np


def load_data(train_start_date, train_end_date, test_start_date, test_end_date, x_length, code):
    '''
    지정한 기간 데이터 가져오기
    :param datetime train_start_date: 시작일
    :param datetime train_end_date: 종료일
    :param datetime train_start_date: 시작일
    :param datetime train_end_date: 종료일
    :param int x_length: x_train의 시퀀스 길이
    :param str code: 종목코드
    :return: ((x_train, y_train), (x_test, y_test))
    '''
    df_train = mu.marcap_date_range(train_start_date, train_end_date, code)
    date_list_train = df_train['Date'].unique()
    total_x_train = len(date_list_train) - x_length
    train_data = [([df_train.at[i + j, 'Close'] for j in range(x_length)], df_train.at[i + x_length, 'Close']) for i
                  in range(total_x_train)]

    df_test = mu.marcap_date_range(test_start_date, test_end_date, code)
    date_list_test = df_test['Date'].unique()
    total_x_test = len(date_list_test) - x_length
    test_data = [([df_test.at[i + j, 'Close'] for j in range(x_length)], df_test.at[i + x_length, 'Close']) for i
                  in range(total_x_test)]

    X_train = np.array([X for (X, y) in train_data])
    X_test = np.array([X for (X, y) in test_data])
    y_train = np.array([y for (X, y) in train_data])
    y_test = np.array([y for (X, y) in test_data])

    return (X_train, y_train), (X_test, y_test)


