import pandas as pd
import numpy as np
import itertools
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def get_closest_values(values, x):
    a = values - x
    lower_value = np.where(a < 0, a, -np.inf).argmax()
    upper_value = np.where(a > 0, a, np.inf).argmin()

    return values[lower_value], values[upper_value]

# Will not work if :
# x[0] < 220 or x[0] > 260
# x[1] < 0 or x[1] > 600
# x[2] < 10 or x[2] > 60
def get_cube(x, data):
    a = get_closest_values(data["Melt temp"].unique(), x[0])
    b = get_closest_values(data["Pack press"].unique(), x[1])
    c = get_closest_values(data["Injection speed"].unique(), x[2])

    # Create cartesian product (A X B)
    return list(itertools.product(a, b, c))

def predict(x, data):
    # data_unique_x = np.vstack({tuple(d) for d in data.to_numpy()[:, :3]})
    # neigh = NearestNeighbors(n_neighbors=8, algorithm="brute", metric="euclidean")
    # neigh.fit(data_unique_x)
    # neighbors_idx = neigh.kneighbors([x], return_distance=False)[0]
    # test = data_unique_x[neighbors_idx]

    cube = np.asarray(get_cube(x, data))
    y = []
    for c in cube:
        cluster = data.loc[
            (data["Melt temp"] == c[0]) &
            (data["Pack press"] == c[1]) &
            (data["Injection speed"] == c[2])]

        median_weight = np.median(cluster["weight"])
        median_width = np.median(cluster["width"])

        y.append([median_weight, median_width])

    lm = linear_model.LinearRegression()
    model = lm.fit(cube, y)

    return model.predict([x])


# predict([249, 250, 30], data_abs)