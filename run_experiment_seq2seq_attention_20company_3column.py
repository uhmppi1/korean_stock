from dataset import data_loader_k200_serial_scaled as data_loader
from model import attention_with_code_embedding
import numpy as np
import json
import train, infer
import dataset.marcap_utils as mu

# to reproduce same training result
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

if __name__ == "__main__":

    train_start_date = '2018-01-01'
    # train_start_date = '2018-06-01'
    train_end_date = '2018-06-30'
    test_start_date = '2018-07-01'
    # test_end_date = '2018-07-15'
    test_end_date = '2018-08-31'
    x_window_length = 10
    y_window_length = 5
    # company_code_list = ['009150'] # code = '009150' # 삼성전기
    company_code_list = mu.krx200_code_list_major # 20 companies including '009150'

    dataset_pickle_path = 'dataset/pickle/dataset_5step_20company_3col.pkl'

    featured_columns = ['Amount', 'Marcap', 'ForeignRatio']
    lognorm_columns = [False, True, False]

    dataloader = data_loader.S2SDataLoader(dataset_pickle_path)

    (X1_train, X2_train, X3_train, y_train), (X1_test, X2_test, X3_test, y_test) = dataloader.load_data(
        train_start_date, train_end_date, test_start_date, test_end_date, x_window_length, y_window_length, featured_columns,
        lognorm_columns, company_code_list)

    print(X1_train.shape)
    print(X2_train.shape)
    print(X3_train.shape)
    print(y_train.shape)
    print(X1_test.shape)
    print(X2_test.shape)
    print(X3_test.shape)
    print(y_test.shape)

    X_test = [X1_test, X2_test]

    input_data_1 = X1_train
    input_data_2 = X2_train
    target_data = y_train

    teacher_data = np.concatenate((np.expand_dims(input_data_1[:, -1, :], axis=1), target_data[:,:-1,:]), axis=1)

    model = attention_with_code_embedding.get_model(
        hidden_size=50, input_shape=(10, 3), decoder_input_shape=(5,3),
        code_num=20, embed_size=2, output_dim=3)

    model = train.train_seq2seq_model(model, input_data_1, input_data_2, teacher_data, target_data, epochs=200)

    y_hats, y_test_to_compares = infer.infer_testset_with_attention(model, X_test, y_test, y_step=5, index_to_compare=1)

    y_test_invscaled = dataloader.get_invertscaled_values('Marcap', y_test_to_compares)
    y_hats_invscaled = dataloader.get_invertscaled_values('Marcap', y_hats)

    infer.draw_result_plot(y_hats_invscaled, y_test_invscaled)