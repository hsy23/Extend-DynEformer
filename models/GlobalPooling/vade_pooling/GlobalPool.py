import numpy as np
import torch as th

class GlobalPool():
    def __init__(self, trend_class_num, seasonal_class_num):
        self.trend_pool = []
        self.seasonal_pool = []

        self.trend_class_num = trend_class_num
        self.seasonal_class_num = seasonal_class_num

        self.s_init = False
        self.t_init = False

    def build_pool_trend(self, series, classes):
        for i in range(self.trend_class_num):
            cluster_index = np.argwhere(classes == i).squeeze(1)
            if len(cluster_index) != 0:
                self.trend_pool.append(np.average(series[cluster_index], axis=0))
            else:
                self.trend_pool.append(np.zeros(series.shape[1]))
        self.trend_pool = np.asarray(self.trend_pool)
        self.t_init = True

    def build_pool_seasonal(self, series, classes):
        for i in range(self.seasonal_class_num):
            cluster_index = np.argwhere(classes == i).squeeze(1)
            if len(cluster_index) != 0:
                self.seasonal_pool.append(np.average(series[cluster_index], axis=0))
            else:
                # self.seasonal_pool.append(np.zeros(series.shape[1]))
                pass
        self.seasonal_pool = np.asarray(self.seasonal_pool)
        self.s_init = True

    def update_pool_seasonal(self, series, classes, alpha=0.1):
        for i in range(self.seasonal_class_num):
            cluster_index = np.argwhere(classes == i).squeeze(1)
            if len(cluster_index) != 0:
                update_v = np.average(series[cluster_index], axis=0)
                self.seasonal_pool[i] = alpha * update_v + (1 - alpha) * self.seasonal_pool[i]
            else:
                # self.seasonal_pool.append(np.zeros(series.shape[1]))
                pass


class GlobalPoolTorch():
    def __init__(self, class_num, gp_seq_len):
        self.trend_pool = th.empty(class_num, gp_seq_len)
        self.seasonal_pool = th.empty(class_num, gp_seq_len)

        self.trend_class_num = class_num
        self.seasonal_class_num = class_num

        self.s_init = False
        self.t_init = False

    def build_pool_trend(self, series, classes):
        for i in range(self.trend_class_num):
            cluster_index = np.argwhere(classes == i).squeeze(1)
            if len(cluster_index) != 0:
                self.trend_pool[i] = th.mean(series[cluster_index], dim=0)
            else:
                pass
        self.trend_pool = np.asarray(self.trend_pool)
        self.t_init = True

    def build_pool_seasonal(self, series, classes):
        for i in range(self.seasonal_class_num):
            cluster_index = th.argwhere(classes == i).squeeze(1)
            if len(cluster_index) != 0:
                self.seasonal_pool[i] = th.mean(series[cluster_index], dim=0)
            else:
                # self.seasonal_pool.append(np.zeros(series.shape[1]))
                pass
        self.s_init = True

    def update_pool_seasonal(self, series, classes, alpha=0.1):
        for i in range(self.seasonal_class_num):
            cluster_index = th.argwhere(classes == i).squeeze(1)
            if len(cluster_index) != 0:
                update_v = th.mean(series[cluster_index], dim=0)
                self.seasonal_pool[i] = alpha * update_v + (1 - alpha) * self.seasonal_pool[i]
            else:
                # self.seasonal_pool.append(np.zeros(series.shape[1]))
                pass