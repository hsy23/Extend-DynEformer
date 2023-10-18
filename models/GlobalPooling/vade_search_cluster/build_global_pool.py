import pickle
import matplotlib.pyplot as plt
from models.GlobalPooling.vade_pooling.GlobalPool import GlobalPool


def draw_series(s):
    plt.plot(s, label='s')
    plt.xlabel('t')
    plt.ylabel('workload_minmax')
    # plt.title('trend clusters valuation')
    plt.show()

def build_gp(decomp_X, seasonal_cluster_res, args, cluster_num):
    gp = GlobalPool(cluster_num, cluster_num)
    gp.build_pool_seasonal(decomp_X, seasonal_cluster_res)

    pickle.dump(gp, open('../pools/pools_0809_sx_s12_big/global_pool_c{}_s{}_s{}.pkl'.format(cluster_num, args.series_len, args.step), 'wb'))


if __name__ == "__main__":
    print('test')