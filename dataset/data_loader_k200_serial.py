import dataset.marcap_utils as mu
import numpy as np
import pandas as pd
import pickle
import os

pickle_file_path_k200_serial = 'dataset/pickle/dataset_k200_serial.pkl'

def load_data(train_start_date, train_end_date, test_start_date, test_end_date, x_length, featured_columns, lognorm_columns, load_pickle=True):
    '''
    지정한 기간 데이터 가져오기
    :param datetime train_start_date: 시작일
    :param datetime train_end_date: 종료일
    :param datetime train_start_date: 시작일
    :param datetime train_end_date: 종료일,
    :param int x_length: x_train의 시퀀스 길이
    :param list code: 종목코드리스트
    :param boolean load_pickle : 저장된 데이터셋 파일을 로딩할지 여부(False면 무조건 새로 생성)
    :return: ((x1_train, x2_train, x3_train, y_train), (x1_test, x2_test, x3_test, y_test))
            x1 : x_length 길이만큼의 이전 시퀀스 데이터, x2 : 종목코드, x3 : K200 종목들의 직전일 시총  y : 예측일의 데이터
    '''
    train_start = np.datetime64(train_start_date)
    train_end = np.datetime64(train_end_date)
    test_start = np.datetime64(test_start_date)
    test_end = np.datetime64(test_end_date)

    X1_train = []
    X2_train = []
    X3_train = []
    y_train = []
    X1_test = []
    X2_test = []
    X3_test = []
    y_test = []

    # step 0 : 혹시 이미 저장되어 있는 데이터셋 오브젝트가 있으면 로딩하여 리턴
    if load_pickle and os.path.exists(pickle_file_path_k200_serial):
        with open(pickle_file_path_k200_serial, 'rb') as file:
            dataset_obj = pickle.load(file)
            (X1_train, X2_train, X3_train, y_train), (X1_test, X2_test, X3_test, y_test) = dataset_obj
    else:
        # step 1 : code-date 기준으로 전체데이터 pivot table을 만든다. (최장 30초 이내 만들어짐)
        df_k200 = mu.marcap_date_range(train_start_date, test_end_date, mu.krx200_code_list)
        df_k200_pv = pd.pivot_table(df_k200, index='Code', columns='Date', values=featured_columns)
        print('df_k200_pv shape : ' , df_k200_pv.shape)

        # step 2 : 전체 일자 기준으로 (x_length + 1) 길이의 시퀀스를 code별로 추출한다.
        # date_list에는 train, test 기간 전체가 다 포함되어 있다.
        date_list = df_k200['Date'].unique()

        # 일단 다 추출해 놓고 나중에 train set에 넣을지 test set에 넣을지 판단할 것이다.
        total_x_train = len(date_list) - x_length
        for i, date in enumerate(date_list):
            for_test = False
            if i >= total_x_train:  break
            print('date : ', date)
            # step 2-1: 먼저 판단을 하자. 이 데이터가 train용인지, test용인지 날짜를 기준으로!!
            predict_date = date_list[i + x_length]
            if predict_date <= train_end:  # 명확히 train data!!
                for_test = False
                print('date : %s for train' % predict_date)
            elif predict_date >= test_start:  # 명확히 test data!!
                for_test = True
                print('date : %s for test' % predict_date)
            else:  # 예측일자가 train_end와 test_start 사이에 끼이면 사용불가
                print('date : %s invalidate' % predict_date)
                continue   # 다음 날짜로 넘어가자.

            for code in mu.krx200_code_list:
                try:
                    # step 2-2 : x1 컬럼들에 대해 필요하면 log transform을 시도.
                    # 만약 데이터에 NaN이 있으면 log(NaN)을 시도하게 되므로 exception이 발생할 것이다.
                    # 여기서 NaN 데이터가 자동으로 걸러지게 된다.
                    x1 = [ [ np.log(df_k200_pv[featured_column][date_list[i + j]][code]) if lognorm_columns[col_idx]
                             else df_k200_pv[featured_column][date_list[i + j]][code]
                             for col_idx, featured_column in enumerate(featured_columns) ]
                            for j in range(x_length) ]
                    x2 = mu.code2index[code]
                    x3 = [ np.log(df_k200_pv['Marcap'][date_list[i + x_length - 1]][c]) for c in mu.krx200_code_list ]
                    y = np.log(df_k200_pv['Marcap'][date_list[i + x_length]][code])
                    print(x1)
                    print(x2)
                    print(x3)
                    print(y)

                    if for_test:  # 명확히 test data!!
                        X1_test.append(x1)
                        X2_test.append(x2)
                        X3_test.append(x3)
                        y_test.append(y)
                    else:  # 명확히 train data!!
                        X1_train.append(x1)
                        X2_train.append(x2)
                        X3_train.append(x3)
                        y_train.append(y)
                except Exception as e:
                    print(type(e))
                    print(e)
                    print('NaN Data for code %s, date %s' % (code, date))
                    continue

        # step 3 : 구축된 dataset은 직렬화해서 pickle로 저장해 두자.
        dataset_obj = ((X1_train, X2_train, X3_train, y_train), (X1_test, X2_test, X3_test, y_test))
        with open('pickle_file_path_k200_serial', 'wb') as file:
            pickle.dump(dataset_obj, file)


    X1_train = np.array(X1_train)
    X2_train = np.array(X2_train)
    X3_train = np.array(X3_train)
    y_train = np.array(y_train)
    X1_test = np.array(X1_test)
    X2_test = np.array(X2_test)
    X3_test = np.array(X3_test)
    y_test = np.array(y_test)

    return (X1_train, X2_train, X3_train, y_train), (X1_test, X2_test, X3_test, y_test)
