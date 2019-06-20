import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


def infer_testset(model, X_test, y_test, input_shape=(10, 1)):
    y_hats = []
    test_size = X_test.shape[0]
    for i in range(test_size):
        x_input = X_test[i]
        x_input_reshape = x_input.reshape(1, input_shape[0], input_shape[1])
        cur_y_hat = model.predict(x_input_reshape, verbose=0)
        cur_y_hat = np.squeeze(cur_y_hat)
        print('TEST DATA %d : predicted=%f, ground_truth=%f' % (i, cur_y_hat, y_test[i]))
        y_hats.append(cur_y_hat)

    return y_hats


def infer_testset_with_embeddingmodel(model, X_test, y_test, input_shape=(10, 1)):
    y_hats = []
    test_size = X_test[0].shape[0]   # X1_test의 길이
    for i in range(test_size):
        x_input = [np.expand_dims(X_test[j][i], axis=0) for j in range(len(X_test))]
        cur_y_hat = model.predict(x_input, verbose=0)
        cur_y_hat = np.squeeze(cur_y_hat)
        print('TEST DATA %d : predicted=%f, ground_truth=%f' % (i, cur_y_hat, y_test[i]))
        y_hats.append(cur_y_hat)

    return y_hats


def infer_testset_with_embeddingmodel_multistep(model, X_test, y_test, input_shape=(10, 3), y_step=5, index_to_compare=1):
    y_hats = []
    y_test_to_compares = []
    test_size = X_test[0].shape[0]   # X1_test의 길이
    for i in range(test_size):
        x_input = [np.expand_dims(X_test[j][i], axis=0) for j in range(len(X_test))]
        cur_y_hat = model.predict(x_input, verbose=0)

        for k in range(1, y_step):
            prev_x_input_0 = x_input[0]    # shape=(1, 10, 3)
            prev_x_input_0 = np.delete(prev_x_input_0, 0, axis=1)  # shape=(1, 9, 3)
            new_x_intput_seq = np.expand_dims(cur_y_hat, axis=0)   # shape=(1, 1, 3)
            cur_x_input_0 = np.concatenate((prev_x_input_0, new_x_intput_seq), axis=1)  # shape=(1, 10, 3)
            x_input[0] = cur_x_input_0
            cur_y_hat = model.predict(x_input, verbose=0)


        cur_y_hat = np.squeeze(cur_y_hat)
        cur_y_hat = cur_y_hat[index_to_compare]
        y_test_to_compare = y_test[i][y_step-1][index_to_compare]

        print('TEST DATA %d : predicted=%f, ground_truth=%f' % (i, cur_y_hat, y_test_to_compare))
        y_hats.append(cur_y_hat)
        y_test_to_compares.append(y_test_to_compare)

    return y_hats, y_test_to_compares


def predict_seq2seq_steps(encoder_model, decoder_model, input_sequence, input_code, output_dim=3, y_step=5):

    input_sequence = np.expand_dims(input_sequence, axis=0)
    input_code = np.expand_dims(input_code, axis=0)
    [_, sh, sc] = encoder_model.predict([input_sequence, input_code])

    i = 0
    start_vec = np.squeeze(input_sequence, axis=0)[-1, :]

    cur_vec = np.zeros((1, output_dim))
    cur_vec[0, :] = start_vec

    output_sequence = []

    for i in range(y_step):
        x_in = [cur_vec, sh, sc]
        [new_vec, sh, sc] = decoder_model.predict(x_in)
        cur_vec[0, :] = new_vec[0, :]
        output_sequence.append(new_vec[0, :])

    return output_sequence


def infer_testset_with_seq2seq(encoder_model, decoder_model, X_test, y_test, y_step=5, index_to_compare=1):
    y_hats = []
    y_test_to_compares = []
    test_size = X_test[0].shape[0]   # X1_test의 길이
    print('test_size :', test_size)
    for i in range(test_size):
        input_sequence = X_test[0][i, :, :]
        print(input_sequence.shape)
        input_code = X_test[1][i]
        output_sequence = predict_seq2seq_steps(encoder_model, decoder_model, input_sequence, input_code, output_dim=3, y_step=5)

        cur_y_hat = np.squeeze(output_sequence[-1, :])
        cur_y_hat = cur_y_hat[index_to_compare]
        y_test_to_compare = y_test[i][y_step-1][index_to_compare]

        print('TEST DATA %d : predicted=%f, ground_truth=%f' % (i, cur_y_hat, y_test_to_compare))
        y_hats.append(cur_y_hat)
        y_test_to_compares.append(y_test_to_compare)

    return y_hats, y_test_to_compares


def draw_result_plot(y_hats, y_test):
    y_hats = np.array(y_hats)
    y_test = np.array(y_test)
    plt.plot(y_test)
    plt.plot(y_hats)
    plt.title('RMSE: %.4f' % np.sqrt(sum((y_hats - y_test) ** 2) / len(y_test)))
    plt.show(block=True)