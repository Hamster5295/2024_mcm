# Feature extracting
# Variables with 'x_' prefix are features to be studied
import mpmath
import numpy as np
from sklearn.preprocessing import Normalizer, OrdinalEncoder, MinMaxScaler
from sklearn.svm import SVR


def extract_features(data, clean=True):
    """
    Extract features from given data

    :param clean: Whether to clean the data with sigma tests.
    :param data: The original dataset
    :return: Extracted features, return as (features, labels)
    """

    x_deltatime = np.zeros(data.shape[0])
    x_total_time = np.zeros(data.shape[0])
    x_total_distance = np.zeros(data.shape[0])
    x_combo = np.zeros(data.shape[0])
    x_is_final = data[:, 5] == 13
    x_is_final = x_is_final.astype(int)
    x_last_pointed = np.zeros(data.shape[0])
    x_unf_err = np.zeros(data.shape[0])
    x_serve_err = np.zeros(data.shape[0])
    x_unf_err_density = np.zeros(data.shape[0])
    x_unf_err_density_p2 = np.zeros(data.shape[0])
    x_serve_err_density = np.zeros(data.shape[0])
    x_serve_err_density_p2 = np.zeros(data.shape[0])

    x_ace_cnt = np.zeros(data.shape[0])
    x_ace_density = np.zeros(data.shape[0])
    x_ace_density_p2 = np.zeros(data.shape[0])
    x_winner_density = np.zeros(data.shape[0])
    x_winner_density_p2 = np.zeros(data.shape[0])
    x_net_density = np.zeros(data.shape[0])
    x_net_density_p2 = np.zeros(data.shape[0])

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

        i = 0
        for i in range(1, 5):
            if t - i < 0 or data[t - i, 5] != data[t, 5]:
                i -= 1
                break
        start_idx = t - i
        deltatime = x_total_time[t] - x_total_time[start_idx]
        if deltatime == 0:
            x_unf_err_density[t] = 0
            x_unf_err_density_p2[t] = 0
            x_serve_err_density[t] = 0
            x_serve_err_density_p2[t] = 0
            x_ace_density[t] = 0
            x_ace_density_p2[t] = 0
            x_winner_density[t] = 0
            x_winner_density_p2[t] = 0
        else:
            x_unf_err_density[t] = np.sum(data[start_idx:t + 1, 27]) / deltatime
            x_unf_err_density_p2[t] = np.sum(data[start_idx:t + 1, 28]) / deltatime

            x_serve_err_density[t] = np.sum(data[start_idx:t + 1, 25]) / deltatime
            x_serve_err_density_p2[t] = np.sum(data[start_idx:t + 1, 26]) / deltatime

            x_ace_density[t] = np.sum(data[start_idx:t + 1, 20]) / deltatime
            x_ace_density_p2[t] = np.sum(data[start_idx:t + 1, 21]) / deltatime

            x_winner_density[t] = np.sum(data[start_idx:t + 1, 22] - x_ace_density[t]) / deltatime
            x_winner_density_p2[t] = np.sum(data[start_idx:t + 1, 23] - x_ace_density_p2[t]) / deltatime

            x_net_density[t] = np.sum(data[start_idx:t + 1, 29]) / deltatime
            x_net_density_p2[t] = np.sum(data[start_idx:t + 1, 30]) / deltatime

        if data[t, 15] == 1:
            combo_cnt = 0
            while data[t - 1 - combo_cnt, 15] == 1 and t - 1 - combo_cnt >= game_idx:
                combo_cnt += 1
            x_combo[t] = combo_cnt

        x_ace_cnt[t] = np.sum(data[set_idx:t, 20])

        is_last = t == data.shape[0] - 1
        if (x_total_time[t] == 0 and t > 0) or is_last:
            cur_idx = t if not is_last else t + 1

            x_deltatime[set_idx:cur_idx] = np.hstack((0, np.diff(x_total_time[set_idx:cur_idx])))
            x_total_distance[set_idx:cur_idx] = np.cumsum(data[set_idx:cur_idx, 39])

            set_idx = cur_idx

        x_serve_err[t] = np.sum(data[game_idx:t, 25])
        x_unf_err[t] = np.sum(data[game_idx:t, 27])

    x_point_subs = data[:, 11:13].astype(str)
    x_point_subs[x_point_subs == '15'] = '1'
    x_point_subs[x_point_subs == '30'] = '2'
    x_point_subs[x_point_subs == '40'] = '3'
    x_point_subs[x_point_subs == 'AD'] = '4'  # We only consider the subtraction
    x_point_subs = x_point_subs.astype(int)
    x_point_subs = x_point_subs[:, 0] - x_point_subs[:, 1]

    x_server = (data[:, 13] == 1).astype(int) * 100

    def remap(arr):
        return OrdinalEncoder().fit_transform(arr.astype(str).reshape(-1, 1)).astype(int)

    x_techniques = np.hstack((data[:, 20:24], remap(data[:, 24]), data[:, 25:40]))
    x_serve_conditions = np.hstack((data[:, 41:43], remap(data[:, 43]), remap(data[:, 44]), remap(data[:, 45])))

    # Cleaning: Unexpected delta times
    def sigma_test(arr):
        ave = np.average(arr)
        std = np.std(arr)
        return np.abs(arr - ave) < 3 * std

    filter_mask = sigma_test(x_deltatime) | (not clean)

    features = np.hstack((
        x_deltatime.reshape(-1, 1),  # Delta time - 0
        x_total_time.reshape(-1, 1),  # Total time of a game - 1
        data[:, 16:18],  # Points earned by each player - 2~3
        x_total_distance.reshape(-1, 1),  # Total distance each player travelled - 4
        x_point_subs.reshape(-1, 1),  # The current difference in points - 5
        x_combo.reshape(-1, 1),  # Combo count - 6
        x_last_pointed.reshape(-1, 1),  # Did the player gain his point last game? - 7
        x_unf_err.reshape(-1, 1),  # The unforced error player 1 has made - 8
        x_serve_err.reshape(-1, 1),  # Double fault when serving - 9
        x_is_final.reshape(-1, 1),  # Was the game a tie-breaker? - 10
        x_server.reshape(-1, 1),  # Is Player 1 the server? - 11
        x_serve_conditions,  # The serving conditions - 12~16
        x_techniques,  # Technique details - 17~36
        x_unf_err_density.reshape(-1, 1),  # The average unforced error density by time - 37
        x_serve_err_density.reshape(-1, 1),  # The average serving error density by time - 38
        x_ace_cnt.reshape(-1, 1),  # Ace count in a set - 39
        x_ace_density.reshape(-1, 1),  # Ace density - 40
        x_winner_density.reshape(-1, 1),  # Winner density (Ace excluded) - 41
        x_net_density.reshape(-1, 1),  # Net density - 42
        x_unf_err_density_p2.reshape(-1, 1),  # Player2's unforced error density - 43
        x_serve_err_density_p2.reshape(-1, 1),  # Player2's serving error density - 44
        x_ace_density_p2.reshape(-1, 1),  # Player2's Ace density - 45
        x_winner_density_p2.reshape(-1, 1),  # Player2's Winner density (Ace excluded) - 46
        x_net_density_p2.reshape(-1, 1),  # Player2's Net density - 47
    )
    )[filter_mask]
    labels = (data[:, 15][filter_mask] == 1).astype(int).astype(str)

    return features, labels


def extract_features_classifier(data, clean=True):
    """
    Extract data for classifier model

    :param data: The original dataset
    :return: Extracted features, return as (features, labels)
    """
    features, label = extract_features(data, clean)
    return MinMaxScaler().fit_transform(features[:, :39]), label


def extract_features_althletes(data, athletes):
    """
    Extract features for logistic fitness
    Features extracted are organized by individual athletes

    :param data: The original dataset
    :param athletes: The athletes required for extraction
    :return: Features extracted
                The first returning value is a dictionary organized as {'Name': features}
                The second is the whole feature sheet with labels as the last column
    """

    def norm(arr):
        return (arr - arr.min()) / (arr.max() - arr.min())

    feature_all, _ = extract_features(data, False)
    ace_density = norm(feature_all[:, 40])
    unf_err_density = norm(feature_all[:, 37])

    data_dict = {}
    for player in athletes:
        matches_p1 = data[data[:, 1] == player]
        matches_p2 = data[data[:, 2] == player]

        set_cnt = np.unique(matches_p1[:, 0] + matches_p1[:, 4].astype(str)).size
        set_cnt += np.unique(matches_p2[:, 0] + matches_p2[:, 4].astype(str)).size

        combos_temp = np.zeros((matches_p1.shape[0] + matches_p2.shape[0], 2))

        x_combos = np.zeros((set_cnt, 2))
        x_serve_point = np.zeros((set_cnt, 2))

        x_ace_cnt = np.zeros(set_cnt)
        x_unf_err_cnt = np.zeros(set_cnt)

        y_victor = np.zeros(set_cnt)

        set_i = 0
        if matches_p1.size != 0:
            set_no = matches_p1[0, 4]
            set_idx = 0
            game_no = matches_p1[0, 5]

            for i in range(matches_p1.shape[0]):
                if matches_p1[i, 5] != game_no or i == matches_p1.shape[0] - 1:
                    game_no = matches_p1[i, 5]

                if matches_p1[i, 4] != set_no or i == matches_p1.shape[0] - 1:
                    # 按“盘”结算
                    x_combos[set_i, 0] = np.max(combos_temp[set_idx:i, 0])
                    x_combos[set_i, 1] = np.min(combos_temp[set_idx:i, 1])

                    set_coll = matches_p1[set_idx:i, :]
                    x_serve_point[set_i, 0] = np.sum((set_coll[:, 13] == 1) & (set_coll[:, 15] == 1))
                    x_serve_point[set_i, 1] = np.sum((set_coll[:, 13] == 2) & (set_coll[:, 15] == 1))

                    y_victor[set_i] = 1 if matches_p1[i if i == matches_p1.shape[0] - 1 else i - 1, 19] == 1 else 0

                    x_ace_cnt[set_i] = np.sum(ace_density[set_idx:i] > 0.05)
                    x_unf_err_cnt[set_i] = np.sum(unf_err_density[set_idx:i] > 0.05)
                    # x_ace_cnt[set_i] = np.sum(ace_density[set_idx:i])
                    # x_unf_err_cnt[set_i] = np.sum(unf_err_density[set_idx:i])

                    set_i += 1
                    set_no = matches_p1[i, 4]
                    set_idx = i

                # 连续得/失分
                combo_cnt = 0
                while i - 1 - combo_cnt >= 0 \
                        and matches_p1[i - 1 - combo_cnt, 15] == matches_p1[i, 15] \
                        and matches_p1[i - 1 - combo_cnt, 5] == game_no:
                    combo_cnt += 1
                if matches_p1[i, 15] == 1:
                    combos_temp[i, 0] = combo_cnt
                else:
                    combos_temp[i, 1] = -combo_cnt

        if matches_p2.size != 0:
            matches_p1 = matches_p2

            offset = set_i
            set_no = matches_p1[0, 4]
            set_idx = 0
            game_no = matches_p1[0, 5]

            for i in range(matches_p1.shape[0]):
                if matches_p1[i, 5] != game_no or i == matches_p1.shape[0] - 1:
                    game_no = matches_p1[i, 5]

                if matches_p1[i, 4] != set_no or i == matches_p1.shape[0] - 1:
                    # 按“盘”结算
                    x_combos[set_i, 0] = np.max(combos_temp[set_idx:i, 0])
                    x_combos[set_i, 1] = np.min(combos_temp[set_idx:i, 1])

                    set_coll = matches_p1[set_idx:i, :]
                    x_serve_point[set_i, 0] = np.sum((set_coll[:, 13] == 2) & (set_coll[:, 15] == 2))
                    x_serve_point[set_i, 1] = np.sum((set_coll[:, 13] == 1) & (set_coll[:, 15] == 2))

                    y_victor[set_i] = 1 if matches_p1[i if i == matches_p1.shape[0] - 1 else i - 1, 19] == 2 else 0

                    x_ace_cnt[set_i] = np.sum(ace_density[set_idx:i] > 0.05)
                    x_unf_err_cnt[set_i] = np.sum(unf_err_density[set_idx:i] > 0.05)
                    # x_ace_cnt[set_i] = np.sum(ace_density[set_idx:i])
                    # x_unf_err_cnt[set_i] = np.sum(unf_err_density[set_idx:i])

                    set_i += 1
                    set_no = matches_p1[i, 4]
                    set_idx = i

                # 连续得/失分
                combo_cnt = 0
                while i - 1 - combo_cnt >= 0 \
                        and matches_p1[i - 1 - combo_cnt, 15] == matches_p1[i, 15] \
                        and matches_p1[i - 1 - combo_cnt, 5] == game_no:
                    combo_cnt += 1
                if matches_p1[i, 15] == 2:
                    combos_temp[i + offset, 0] = combo_cnt
                else:
                    combos_temp[i + offset, 1] = -combo_cnt

        data_cur = np.hstack((x_combos[:, 0:1], x_serve_point, x_ace_cnt.reshape(-1, 1), x_unf_err_cnt.reshape(-1, 1)))
        data_dict[player] = np.hstack((data_cur, y_victor.reshape(-1, 1)))

    data_sheet = None
    for _, item in data_dict.items():
        if data_sheet is None:
            data_sheet = item
        else:
            data_sheet = np.vstack((data_sheet, item))

    return data_dict, data_sheet


def extract_features_nn(data, p2=False):
    def norm(arr):
        divider = arr.max() - arr.min()
        return (arr - arr.min()) / divider if divider != 0 else 1

    def get_ave_speed(data, feature, smoothing=16):
        ave_speed = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            start_idx = max(0, i - smoothing)
            if i - start_idx == 0:
                ave_speed[i] = 0
            else:
                ave_speed[i] = np.sum(data[start_idx:i + 1, 39]) / (
                        feature[i, 1] - feature[start_idx, 1]) if not p2 \
                    else np.sum(data[start_idx:i + 1, 40]) / (feature[i, 1] - feature[start_idx, 1])
        ave_speed[0] = ave_speed[1]
        return norm(ave_speed)

    def regress_ave_speed(ave_speed):
        xs = np.arange(ave_speed.size).reshape(-1, 1)
        ave_speed_regressed = norm(SVR().fit(xs, norm(ave_speed)).predict(xs))
        return ave_speed_regressed

    def regress_momentum(features):
        showoffs = norm(np.sum(features[:, 40:43], axis=1)) - norm(np.sum(features[:, 37:39], axis=1)) if not p2 \
            else norm(np.sum(features[:, 45:48], axis=1)) - norm(np.sum(features[:, 43:45], axis=1))
        xs = np.where(np.abs(showoffs) > 0.2)[0].reshape(-1, 1)
        momentum_features = showoffs[xs].reshape(-1)

        weights = np.copy(momentum_features)
        weights[np.abs(weights) < 0.2] = 0
        weights[(np.abs(weights) >= 0.2) & (np.abs(weights) < 0.5)] = 1
        weights[np.abs(weights) > 0.5] = 2
        model = SVR().fit(xs, momentum_features, weights)
        return showoffs, model.predict(np.arange(features.shape[0]).reshape(-1, 1))

    features, labels = extract_features(data, False)
    matches = np.unique(data[:, 0])

    x_ave_speeds = np.zeros((features.shape[0], 2))
    x_show_offs = np.zeros(features.shape[0])
    y_momentum_labels = np.zeros(features.shape[0])

    for match in matches:
        match_filter = data[:, 0] == match
        match_data = data[match_filter]
        match_feature = features[match_filter]

        ave_speed = get_ave_speed(match_data, match_feature)
        ave_speed_regressed = regress_ave_speed(ave_speed)
        x_ave_speeds[match_filter, 0] = ave_speed
        x_ave_speeds[match_filter, 1] = ave_speed_regressed
        x_show_offs[match_filter], y_momentum_labels[match_filter] = regress_momentum(match_feature)

    return np.hstack((
        x_ave_speeds,
        x_show_offs.reshape(-1, 1)
    )), y_momentum_labels
