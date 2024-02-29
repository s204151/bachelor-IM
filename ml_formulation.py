from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import StandardScaler


class f_theta:

    def __init__(self, data, L, delta=0.01):
        self.delta = delta
        y = data[:, -2:]
        self.y_star = [np.mean(y[:, 0]), np.mean(y[:, 1])]

        # Transform data so rows include lagged observations
        transformed_data = []
        for i in range(1, data.shape[0] - L + 1):
            z = [data[- i - j] for j in range(L, -1, -1)]
            transformed_data.append(np.hstack(z))
        transformed_data.reverse()
        transformed_data = np.asarray(transformed_data)

        epsilons = transformed_data[:, -5:-2] - transformed_data[:, -10:-7]

        data_x = np.hstack([transformed_data[:, 0:-2], epsilons])
        data_y = transformed_data[:, -2:]
        lm = linear_model.LinearRegression()
        self.model = lm.fit(data_x, data_y)

    def predict(self, Z, epsilon):
        return self.model.predict([np.append(Z, epsilon)])

    def epsilon_star(self, Z):
        check_range = np.linspace(-self.delta, self.delta, 10)
        result = []
        for i in check_range:
            for j in check_range:
                for k in check_range:
                    if np.sqrt(i ** 2 + j ** 2 + k ** 2) < self.delta:
                        y = self.predict(Z, [i, j, k])[0]
                        result.append([np.sum(self.y_star - y), self.y_star - y, y, [i, j, k]])

        result = np.asarray(result, dtype=object)
        idx = np.argmin(result[:, 0])
        return result[idx, -1]