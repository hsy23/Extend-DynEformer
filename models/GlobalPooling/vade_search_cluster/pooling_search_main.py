import argparse
from vade_search import VaDE
import numpy as np
from build_global_pool import build_gp
from matplotlib import pyplot as plt
import pickle


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='VaDE')
    parse.add_argument('--batch_size', type=int, default=40000)
    parse.add_argument('--dataX_dir', type=str, default='../../../data/ECW_0809_big.npy')
    parse.add_argument('--nClusters', type=list, default=list(range(900, 1200, 100)))

    parse.add_argument('--series_len', type=int, default=168)
    parse.add_argument('--step', type=int, default=12)

    parse.add_argument('--hid_dim', type=int, default=10)
    parse.add_argument('--cuda', type=bool, default=True)

    args=parse.parse_args()

    X = np.load(open(args.dataX_dir, 'rb'), allow_pickle=True)
    X = np.asarray(X)

    train_len = int(0.8*X.shape[1])
    X_train = X[:, :train_len, 0]

    vade = VaDE(args, X_train)
    if args.cuda:
        vade = vade.cuda()

    X_train_all_np, keep_n, all_pre_res, aic_l, bic_l = vade.train_search(pre_epoch=3000)

    # keep_n = list(range(50, 850, 50)) + list(range(900, 1001, 50))
    # all_pre_res = pickle.load(open('./seasonal_pre_res.pkl', 'rb'))
    # aic_l = pickle.load(open('./seasonal_aic_l.pkl', 'rb'))+pickle.load(open('./seasonal_aic_l_650+.pkl', 'rb'))
    # bic_l = pickle.load(open('./seasonal_bic_l.pkl', 'rb'))+pickle.load(open('./seasonal_bic_l_650+.pkl', 'rb'))

    draw_plot = True
    if draw_plot:
        fig, ax1 = plt.subplots()
        # 画第一条曲线和设置它的 y 轴（左侧）
        ax1.plot(keep_n, aic_l, label='aic', color='b')
        ax1.set_xlabel('n_clusters')
        ax1.set_ylabel('aic', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.legend(loc='upper left')

        # 创建共享 x 轴的新的 y 轴
        ax2 = ax1.twinx()

        # 画第二条曲线和设置它的 y 轴（右侧）
        ax2.plot(keep_n, bic_l, label='bic', color='r')
        ax2.set_ylabel('bic', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_xticks(keep_n)
        ax2.set_xticklabels(keep_n)
        ax2.legend(loc='upper right')

        plt.title('seasonal clusters valuation')
        plt.show()

    for i, pre_res in enumerate(all_pre_res):
        build_gp(X_train_all_np, pre_res, args, keep_n[i])