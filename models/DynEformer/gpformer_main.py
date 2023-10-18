import numpy as np
import pandas as pd
import random
from torch import nn
import argparse
import pickle
from gpformer import gpformer
from sklearn import preprocessing
from tqdm import tqdm
from torch.optim import Adam
import torch
import time
from torch.utils.data import DataLoader, Dataset

from models.global_utils import train_test_split


def get_mape(yTrue, yPred, scaler=None):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)

    return np.mean(np.abs((yTrue - yPred) / yTrue) * 100)


def get_mse(yTrue, yPred, scaler=None):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)
    return np.mean((yTrue - yPred) ** 2)


def get_mae(yTrue, yPred, scaler):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)

    return np.mean(np.abs(yTrue - yPred))


class PPIO_Dataset(Dataset):
    def __init__(self, X, enc_len=48, label_len=12, pred_len=24, step=12):
        num_ts, num_periods, num_features = X.shape
        X_train_all = []
        Y_train_all = []
        X_mark_all = []
        Y_mark_all = []

        for i in range(num_ts):
            for j in range(enc_len, num_periods - pred_len, step):
                X_train_all.append(X[i, j - enc_len:j, 0])
                Y_train_all.append(X[i, j - label_len:j + pred_len, 0])
                X_mark_all.append(X[i, j - enc_len:j, 1:])  # 携带静态特征
                Y_mark_all.append(X[i, j - label_len:j + pred_len, 1:4])

        self.X = np.asarray(X_train_all).reshape(-1, enc_len, 1)
        self.Y = np.asarray(Y_train_all).reshape(-1, label_len + pred_len, 1)
        self.X_mark = np.asarray(X_mark_all).reshape(-1, enc_len, 15)
        self.Y_mark = np.asarray(Y_mark_all).reshape(-1, label_len + pred_len, 3)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.X_mark[index], self.Y_mark[index]


def get_ppio(X, y, batch_size, args):
    Xtr, ytr, Xte, yte = train_test_split(X, y)
    num_ts, num_periods, num_features = Xte.shape

    xscaler = preprocessing.MinMaxScaler()
    yscaler = preprocessing.MinMaxScaler()
    yscaler.fit(Xtr[:, :, 0].reshape(-1, 1))
    Xtr = xscaler.fit_transform(Xtr.reshape(-1, num_features)).reshape(num_ts, -1, num_features)
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts, -1, num_features)

    # pickle.dump([xscaler, yscaler], open('8_scalers.pkl', 'wb'))
    Xtr_loader = DataLoader(PPIO_Dataset(Xtr, args.enc_len, args.label_len, args.pred_len), batch_size=batch_size)
    Xte_loader = DataLoader(PPIO_Dataset(Xte, args.enc_len, args.label_len, args.pred_len), batch_size=batch_size)

    return Xtr_loader, Xte_loader, yscaler


def train(X, y, args):
    device = torch.device('cuda:0')

    model = gpformer(args)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    random.seed(2)

    Xtr_loader, Xte_loader, yscaler = get_ppio(X, y, args.batch_size, args)

    train_loss = []
    test_loss = []
    test_mse = []
    test_mae = []
    test_mape = []
    criterion = nn.MSELoss().to(device)

    min_loss = 1000
    # training
    for epoch in range(args.num_epoches):
        epo_train_losses = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(Xtr_loader):
            optimizer.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_static = batch_x_mark[:, 0, 3:].float().to(device).squeeze(1)

            batch_x_mark = batch_x_mark.float().to(device)[:, :, :3]
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_static, epoch % 50==0)

            f_dim = -1
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)

            epo_train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        train_loss.append(np.mean(epo_train_losses))

        epo_test_losses = []
        epo_mse = []
        epo_mape = []
        epo_mae = []
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(Xte_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_static = batch_x_mark[:, 0, 3:].float().to(device).squeeze(1)

                batch_x_mark = batch_x_mark.float().to(device)[:, :, :3]
                batch_y_mark = batch_y_mark.float().to(device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_static, False)

                f_dim = -1
                outputs = outputs[:, -args.pred_len:, f_dim]
                batch_y = batch_y[:, -args.pred_len:, f_dim]
                loss = criterion(outputs, batch_y)
                epo_test_losses.append(loss.item())
                epo_mse.append(get_mse(outputs.cpu(), batch_y.cpu(), yscaler))
                epo_mape.append(get_mape(outputs.cpu(), batch_y.cpu(), yscaler))
                epo_mae.append(get_mae(outputs.cpu(), batch_y.cpu(), yscaler))

        test_loss.append(np.mean(epo_test_losses))
        test_mse.append(np.mean(epo_mse))
        test_mape.append(np.mean(epo_mape))
        test_mae.append(np.mean(epo_mae))

        if args.save_model:
            if test_loss[-1] < min_loss:
                best_model = model
                min_loss = test_loss[-1]
                torch.save(model, 'saved_model/DynEformer_pro_best_n{}.pt'.format(cn))

        print(f'epoch {epoch}, train loss: {train_loss[-1]}, test loss: {test_loss[-1]}, '
              f'mse: {test_mse[-1]}, mape: {test_mape[-1]}, mae: {test_mae[-1]}')

    print(np.min(test_mse), np.min(test_mape), np.min(test_mae), np.argmin(test_loss), 'convergence point')

    return train_loss, test_loss, test_mse, test_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data process
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--minmax_scaler", "-mm", action="store_true", default=True)

    # train setting
    parser.add_argument("--num_epoches", "-e", type=int, default=200)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=256)

    # model setting
    parser.add_argument("--e_layers", "-nel", type=int, default=2)
    parser.add_argument("--d_layers", "-ndl", type=int, default=1)
    parser.add_argument("--d_model", "-dm", type=int, default=256)  # 嵌入维度
    parser.add_argument("--d_low", "-dlow", type=int, default=10)  # 降维后用于聚类的维度
    parser.add_argument("--n_heads", "-nh", type=int, default=8)  # 注意力头数量
    parser.add_argument("--d_ff", "-hs", type=int, default=256)
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument("--label_len", "-dl", type=int, default=12)  # decoder用到的输入长度
    parser.add_argument("--pred_len", "-ol", type=int, default=24)  # 预测长度
    parser.add_argument("--enc_len", "-not", type=int, default=24*2)  # 输入长度
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-activation", type=str, default='gelu')
    parser.add_argument("-dim_static", type=int, default=12)
    parser.add_argument("-gp_len", type=int, default=650)  # 全局池类别数
    parser.add_argument("-gp_seq_len", type=int, default=48)  # 全局池序列长度
    parser.add_argument("-if_padding", type=bool, default=True)
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument("-output_attention", type=bool, default=False)

    # other settings
    parser.add_argument("--run_test", "-rt", action="store_true", default=True)
    parser.add_argument("--save_model", "-sm", type=bool, default=False)
    parser.add_argument("--load_model", "-lm", type=bool, default=False)
    parser.add_argument("--show_plot", "-sp", type=bool, default=False)

    args = parser.parse_args()

    if args.run_test:
        X_all = np.load(open(r"../../data/ECW_0809_big.npy", 'rb'), allow_pickle=True)
        y_all = X_all[:, :, 0]
        losses, test_losses, mse_l, mae_l = train(X_all, y_all, args)

        # 20230901-DynEformer-gp: 0.06530844436197054 27.852698285546992 0.13399404697089226 105 convergence point