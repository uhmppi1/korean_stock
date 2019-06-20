import dataset.marcap_utils as mu
import numpy as np
import pandas as pd


def load_data(train_start_date, train_end_date, test_start_date, test_end_date, x_length, featured_columns, normalize_columns):
    '''
    지정한 기간 데이터 가져오기
    :param datetime train_start_date: 시작일
    :param datetime train_end_date: 종료일
    :param datetime train_start_date: 시작일
    :param datetime train_end_date: 종료일
    :param int x_length: x_train의 시퀀스 길이
    :param list code: 종목코드리스트
    :return: ((x_train, y_train), (x_test, y_test))
    '''
    df_k200 = mu.marcap_date_range(train_start_date, test_end_date, mu.krx200_code_list)
    df_k200_pv = pd.pivot_table(df_k200, index='Code', columns='Date', values=featured_columns)
    print('df_k200_pv shape : ' , df_k200_pv.shape)

    df_train = mu.marcap_date_range(train_start_date, train_end_date, mu.krx200_code_list)
    date_list_train = df_train['Date'].unique()
    total_x_train = len(date_list_train) - x_length

    print('total_x_train : ', total_x_train)

    # iteration : i -> 반복되는 일자 기준
    # j -> windows_size (x_train 시퀀스 길이)
    # code -> 매일별 업체 기준 (code 179건)
    # featured_column -> 추출 컬럼 종류 (featured_columns 반복수)
    train_data = [(
        (
            [
                [ df_k200_pv[featured_column][date_list_train[i + j]][code]
                    for col_idx, featured_column in enumerate(featured_columns) ]
                for j in range(x_length)
            ], mu.code2index[code]
        ) # X = ([],idx)
        , df_k200_pv['Marcap'][date_list_train[i + x_length]][code] # y
    ) # (X,y)
    for code in mu.krx200_code_list for i in range(total_x_train) ]

    print('train_data length : ', len(train_data))

    df_test = mu.marcap_date_range(test_start_date, test_end_date, mu.krx200_code_list)
    date_list_test = df_test['Date'].unique()
    total_x_test = len(date_list_test) - x_length

    print('total_x_test : ', total_x_test)

    test_data = [(
        (
            [
                [ df_k200_pv[featured_column][date_list_test[i + j]][code]
                    for col_idx, featured_column in enumerate(featured_columns) ]
                for j in range(x_length)
            ], mu.code2index[code]
        ) # X = ([],idx)
        , df_k200_pv['Marcap'][date_list_test[i + x_length]][code] # y
    ) # (X,y)
    for code in mu.krx200_code_list for i in range(total_x_test)]

    print('test_data length : ', len(test_data))

    X_train = np.array([X for (X, y) in train_data])
    X_test = np.array([X for (X, y) in test_data])
    y_train = np.array([y for (X, y) in train_data])
    y_test = np.array([y for (X, y) in test_data])

    return (X_train, y_train), (X_test, y_test)


def load_data_ori(train_start_date, train_end_date, test_start_date, test_end_date, x_length):
    '''
    지정한 기간 데이터 가져오기
    :param datetime train_start_date: 시작일
    :param datetime train_end_date: 종료일
    :param datetime train_start_date: 시작일
    :param datetime train_end_date: 종료일
    :param int x_length: x_train의 시퀀스 길이
    :param list code: 종목코드리스트
    :return: ((x_train, y_train), (x_test, y_test))
    '''
    # assert(isinstance(code, list))
    # num_code = len(code)
    df_k200 = mu.marcap_date_range(train_start_date, test_end_date, mu.krx200_code_list)
    df_k200_pv = pd.pivot_table(df_k200, index='Code', columns='Date', values='Marcap')



    df_train = mu.marcap_date_range(train_start_date, train_end_date, mu.krx200_code_list)
    date_list_train = df_train['Date'].unique()
    total_x_train = len(date_list_train) - x_length
    train_data = [([df_train.at[i + j, 'Close'] for j in range(x_length)], df_train.at[i + x_length, 'Close']) for i
                  in range(total_x_train)]

    df_test = mu.marcap_date_range(test_start_date, test_end_date, mu.krx200_code_list)
    date_list_test = df_test['Date'].unique()
    total_x_test = len(date_list_test) - x_length
    test_data = [([df_test.at[i + j, 'Close'] for j in range(x_length)], df_test.at[i + x_length, 'Close']) for i
                 in range(total_x_test)]

    X_train = np.array([X for (X, y) in train_data])
    X_test = np.array([X for (X, y) in test_data])
    y_train = np.array([y for (X, y) in train_data])
    y_test = np.array([y for (X, y) in test_data])

    return (X_train, y_train), (X_test, y_test)


