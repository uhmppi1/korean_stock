import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


def infer_testset(model, X_test, y_test):
    y_hats = []
    cur_y_hat = 0
    test_size = X_test.shape[0]
    for i in range(test_size):
        x_input = X_test[0]
        x_input_reshape = x_input.reshape(1, 10, 1)
        cur_y_hat = model.predict(x_input_reshape, verbose=0)
        cur_y_hat = np.squeeze(cur_y_hat)
        print('TEST DATA %d : predicted=%f, ground_truth=%d' % (i, cur_y_hat, y_test[i]))
        y_hats.append(cur_y_hat)

    return y_hats


def draw_result_plot(y_hats, y_test):
    plt.plot(y_test)
    plt.plot(y_hats)
    plt.title('RMSE: %.4f' % np.sqrt(sum((y_hats - y_test) ** 2) / len(y_test)))