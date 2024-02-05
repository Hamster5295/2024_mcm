import pickle

import numpy as np
import torch.nn as nn
import torch.optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

import feature
import hio
import nn as hnn


class TennisDataset(Dataset):
    def __init__(self, data, p2=False):
        self.features, self.labels = feature.extract_features_nn(data, p2)
        self.features = MinMaxScaler().fit_transform(self.features)

    def __getitem__(self, item):
        return self.features[0:item], self.labels[1:item + 1]

    def __len__(self):
        return self.features.shape[0]


class MomentumModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, output_size=1, layers_cnt=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = nn.GRU(input_size, hidden_size, layers_cnt)
        self.seq = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.seq(x)


data_ori = hio.read_csv('data_ori.csv', skiprows=1)

data_train = data_ori[data_ori[:, 0] != '2023-wimbledon-1701']
dataset_train = TennisDataset(data_train)
dataloader_train = DataLoader(dataset_train, batch_size=64)

data_test = data_ori[data_ori[:, 0] == '2023-wimbledon-1701']
dataset_test = TennisDataset(data_test)
dataset_test_p2 = TennisDataset(data_test, True)
dataloader_test = DataLoader(dataset_test, batch_size=64)

tensor_test = torch.from_numpy(dataset_test.features).to(hnn.device).float()

device = hnn.device

model = MomentumModel(3).to(device)
optim = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

def train():
    epoch = 1

    for i in range(epoch):
        # for batch, X, y in enumerate(dataloader_train):
        for pos in range(1, len(dataset_train) - 1):
            X, y = dataset_train[pos]
            X, y = torch.from_numpy(X).to(device).float(), torch.from_numpy(y).to(device).float()

            pred_p1 = model(X)
            loss = loss_fn(pred_p1.squeeze(), y)

            loss.backward()
            optim.step()
            optim.zero_grad()

            if pos % 1000 == 0:
                print(f"Epoch {i + 1}, Step {pos}/{len(dataset_train) - 1}")
                print(f"Loss: {loss.item()}")
                print()

    print('Done!')

    # hnn.save(model, 'model_lstm.pt')
    hnn.save(model, 'model_gru.pt')

def test():
    model = MomentumModel(3)
    # hnn.load(model, 'model_lstm.pt')
    hnn.load(model, 'model_gru.pt')
    model.cpu()
    model.eval()


    def predict(model, dataset, length=None):
        pred_len = round(len(dataset)) if length is None else length
        pred = model(torch.from_numpy(dataset[pred_len - 1][0]).float()).detach().numpy().squeeze()

        def smooth(pred, extend):
            weight = 1 / extend * np.ones(extend)
            pred_disp = np.convolve(pred, weight)[0:-extend]
            return pred_disp

        pred = pred[10:-1]
        pred = smooth(pred, 4)
        return pred


    def norm(arr):
        return (arr - arr.min()) / (arr.max() - arr.min())


    matches = np.unique(data_ori[:, 0])
    pred_p1s = []
    pred_p2s = []
    for match_name in matches:
        print(match_name)
        match_data = data_ori[data_ori[:, 0] == match_name]
        dataset_p1 = TennisDataset(match_data)
        dataset_p2 = TennisDataset(match_data, True)

        pred_p1 = predict(model, dataset_p1)
        pred_p2 = predict(model, dataset_p2)
        pred_p1 = norm(pred_p1)
        pred_p2 = norm(pred_p2)
        pred_p2 += pred_p1[0] - pred_p2[0]

        k = 1 / pred_p1[pred_p1.nonzero()[0][0]]

        pred_p1 *= k
        pred_p2 *= k

        pred_p1s.append(np.copy(pred_p1))
        pred_p2s.append(np.copy(pred_p2))

    with open('gru_result_p1.npy', 'wb') as file:
        pickle.dump(pred_p1s, file)
    with open('gru_result_p2.npy', 'wb') as file:
        pickle.dump(pred_p2s, file)

train()
test()