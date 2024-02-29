from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np


def calculate_mse(x, y):
    total_y_true = []
    total_y_pred = []
    lm = linear_model.LinearRegression()
    kf = KFold(n_splits=5)
    for train_idx, test_idx in kf.split(x,y):
        model = lm.fit(x[train_idx], y[train_idx])

        total_y_true.extend(y[test_idx])
        total_y_pred.extend(model.predict(x[test_idx]))
    return mean_squared_error(total_y_true, total_y_pred)


def test_lagg_time(data):
    result_width_mse = []
    result_weight_mse = []
    for ell in range(11):
        # print("ell =", ell)

        # Create vector with lagg time
        n_feat = 5
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=(1+ell, n_feat))
        temp_result = []
        for w in windows:
            temp_result.append(np.asarray(w[0]).flatten())
        temp_result = np.asarray(temp_result)

        data_x = temp_result[:, 0:-2]

        data_y = temp_result[:, -2]
        result_1 = calculate_mse(data_x, data_y)
        # print("width mse: ", result_1)
        result_width_mse.append(result_1)

        data_y = temp_result[:,-1]
        result_2 = calculate_mse(data_x, data_y)
        # print("weight mse: ", result_2)
        # print("-----------------------")
        result_weight_mse.append(result_2)

    return result_width_mse, result_weight_mse

def test_lagg_time_doubleY(data):
    results = []
    for ell in range(11):
        # Create vector with lagg time
        n_feat = 5
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=(1+ell, n_feat))
        temp_result = []
        for w in windows:
            temp_result.append(np.asarray(w[0]).flatten())
        temp_result = np.asarray(temp_result)

        data_x = temp_result[:, 0:-2]

        data_y = temp_result[:, -2:]
        result = calculate_mse(data_x, data_y)
        # print("width mse: ", result_1)
        results.append(result)

    return results
