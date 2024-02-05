import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder

import forest
import hio

data = hio.read_csv('data_ori.csv', skiprows=1)

data = data[data[:, 0] != '2023-wimbledon-1701']

# Initial data cleaning
data = data[(data[:, 39] != 0) & (data[:, 40] != 0)]  # Unrecorded distances


# Feature extracting
# Variables with 'x_' prefix are features to be studied
def extract_features(data):
    x_deltatime = np.zeros(data.shape[0])
    x_total_time = np.zeros(data.shape[0])
    x_total_distance = np.zeros(data.shape[0])
    x_combo = np.zeros(data.shape[0])
    x_is_final = data[:, 5] == 13
    x_is_final = x_is_final.astype(int)
    x_last_pointed = np.zeros(data.shape[0])
    x_unf_err = np.zeros(data.shape[0])
    x_serve_err = np.zeros(data.shape[0])

    set_idx = 0
    game_idx = 0
    for t in range(data.shape[0]):
        if data[t, 11] == data[t, 12] and data[t, 11] == '0':
            game_idx = t

        if t != game_idx:
            if data[t - 1, 15] == 1:
                x_last_pointed[t] = 1
        else:
            x_last_pointed[t] = 0

        time_str = data[t, 3]
        hour, minu, sec = time_str.split(':')
        x_total_time[t] = float(hour) % 24 * 3600 + float(minu) * 60 + float(sec)

        if data[t, 15] == 1:
            combo_cnt = 0
            while data[t - 1 - combo_cnt, 15] == 1 and t - 1 - combo_cnt >= game_idx:
                combo_cnt += 1
            x_combo[t] = combo_cnt

        if x_total_time[t] == 0 and t > 0:
            x_deltatime[set_idx:t] = np.hstack((0, np.diff(x_total_time[set_idx:t])))
            x_total_distance[set_idx:t] = np.cumsum(data[set_idx:t, 39])
            set_idx = t

        x_serve_err[t] = np.sum(data[game_idx:t, 25])
        x_unf_err[t] = np.sum(data[game_idx:t, 27])

    x_point_subs = data[:, 11:13].astype(str)
    x_point_subs[x_point_subs == '15'] = '1'
    x_point_subs[x_point_subs == '30'] = '2'
    x_point_subs[x_point_subs == '40'] = '3'
    x_point_subs[x_point_subs == 'AD'] = '4'  # We only consider the subtraction
    x_point_subs = x_point_subs.astype(int)
    x_point_subs = x_point_subs[:, 0] - x_point_subs[:, 1]

    x_server = (data[:, 13] == 1).astype(int)
    x_total_points = data[:, 17:19]

    def remap(arr):
        return OrdinalEncoder().fit_transform(arr.astype(str).reshape(-1, 1)).astype(int)

    x_techniques = np.hstack((data[:, 20:24], remap(data[:, 24]), data[:, 25:40]))
    x_serve_conditions = np.hstack((data[:, 41:43], remap(data[:, 43]), remap(data[:, 44]), remap(data[:, 45])))

    # Cleaning: Unexpected delta times
    def sigma_test(arr):
        ave = np.average(arr)
        std = np.std(arr)
        return np.abs(arr - ave) < 3 * std

    filter_mask = sigma_test(x_deltatime)

    features = np.hstack(
        (x_deltatime.reshape(-1, 1), x_total_time.reshape(-1, 1), data[:, 39:41],
         x_total_distance.reshape(-1, 1),
         x_point_subs.reshape(-1, 1),
         x_total_points, x_combo.reshape(-1, 1),
         x_last_pointed.reshape(-1, 1),
         x_unf_err.reshape(-1, 1), x_serve_err.reshape(-1, 1),
         x_is_final.reshape(-1, 1),
         x_server.reshape(-1, 1), x_serve_conditions, x_techniques))[filter_mask]
    labels = (data[:, 15][filter_mask] == 1).astype(int).astype(str)

    return features, labels


features, labels = extract_features(data)


# PCA analyze
pca = PCA(n_components=0.97)
features = pca.fit_transform(features)

with open('pca.pkl', 'wb') as file:
    pickle.dump(pca, file)

model_rfc = forest.fit_rfc(features, labels)
print()
model_hgbt = forest.fit_hgbtc(features, labels)