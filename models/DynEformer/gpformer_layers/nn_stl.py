import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL


def s_decomp(s, type):
    stl = STL(s, period=24, robust=True)
    res_robust = stl.fit()
    # fig = res_robust.plot()
    # plt.show()
    if type == 'trend':
        return res_robust.trend
    else:
        return res_robust.seasonal


def get_pworkload(X_raw, series_len=24 * 2, step=12, batch_size=256):
    X = X_raw
    if type(X) == list:
        X = np.asarray(X)
    num_ts, num_periods = X.shape

    # std
    scaler = MinMaxScaler()
    new_x = scaler.fit_transform(X.T).T

    # decompose
    Y = []
    print('**********************begin decompose******************************')
    for s in tqdm(new_x):  # If the data requires additional processing for timing decomposition
        Y.append(s_decomp(s, type='seasonal'))
    Y = np.asarray(Y)

    x_all = []
    y_all = []
    for i in range(num_ts):
        for j in range(series_len, num_periods, step):
            x_all.append(new_x[i, j - series_len:j])
            y_all.append(Y[i, j - series_len:j])

    x_all = np.asarray(x_all).reshape(-1, series_len)
    x_all_tensor = torch.from_numpy(x_all).float()

    y_all = np.asarray(y_all)
    y_all_tensor = torch.from_numpy(y_all).float()

    # dataloader = DataLoader(TensorDataset(X_train_all_tensor), batch_size=batch_size, shuffle=True, num_workers=4)
    return x_all_tensor, y_all_tensor


class NNSTL(nn.Module):  # Neural Network STL
    def __init__(self, sequence_length):
        super(NNSTL, self).__init__()

        # Approximate the seasonal component with a 1D conv layer with more filters
        self.seasonal_layer = nn.Conv1d(1, 1, kernel_size=5, padding=2)

        # Add a layer normalization
        self.layer_norm = nn.LayerNorm([1, sequence_length])

        # Approximate the trend component with another 1D conv layer with larger kernel size
        self.trend_layer = nn.Conv1d(1, 1, kernel_size=30, padding=7)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, sequence_length)

        seasonal = self.seasonal_layer(x)
        seasonal = self.layer_norm(seasonal)

        trend = self.trend_layer(seasonal)
        trend = self.layer_norm(trend)

        # Remove channel dimension and return
        return seasonal.squeeze(1), trend.squeeze(1)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


if "__name__" == "__main__":
    # Generate some synthetic data with a trend and seasonal component
    sequence_length = 48
    batch_size = 1

    X = np.load(open('../../../data/ECW_08.npy', 'rb'), allow_pickle=True)

    train_len = int(0.8 * X.shape[1])
    train_data = X[:, :train_len, 0]
    test_data = X[:, train_len:, 0]

    x_train_tensor, y_train_tensor = get_pworkload(train_data)  # x_train_tensor: N*T, y_train_tensor: n*t
    x_test_tensor, y_test_tensor = get_pworkload(test_data)

    # Create and train the model
    model = NNSTL(sequence_length)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Loss function
    mse_loss = nn.MSELoss()

    for epoch in range(1000):
        seasonal, trend = model(x_train_tensor)

        # Loss for the seasonal part
        seasonal_loss = mse_loss(seasonal, y_train_tensor)

        # Reconstruction loss
        recon_loss = mse_loss(x_train_tensor, seasonal + trend)

        # Combined loss
        loss = seasonal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Plotting
    plt.figure(figsize=(8, 6))

    # Take one sample sequence for demonstration
    sample_index = 0
    sample_x = x_test_tensor[sample_index].detach().numpy()
    sample_seasonal, sample_trend = model(x_test_tensor[sample_index].unsqueeze(0))
    sample_seasonal = sample_seasonal.detach().squeeze().numpy()
    sample_trend = sample_trend.detach().squeeze().numpy()

    t = np.linspace(0, sequence_length, sequence_length)

    plt.subplot(4, 1, 1)
    plt.title("Original Sequence")
    plt.plot(t, sample_x)

    plt.subplot(4, 1, 2)
    plt.title("STL Seasonal")
    plt.plot(t, y_test_tensor[sample_index].detach().numpy())

    plt.subplot(4, 1, 3)
    plt.title("Model Seasonal")
    plt.plot(t, sample_seasonal)

    plt.subplot(4, 1, 4)
    plt.title("Model Trend")
    plt.plot(t, sample_trend)

    plt.tight_layout()
    plt.show()
