import matplotlib.pyplot as plt
import pickle
import os.path as osp

class LossPlotter():
    def __init__(self, path, mode='a'):
        self.path = path
        self.mse_loss_list = []
        self.wg_loss_list = []
        self.wc_loss_list = []

        if osp.isfile(osp.join(self.path, 'loss.bin')) and mode == 'a':
            with open(osp.join(self.path, 'loss.bin'), 'rb') as file:
                self.mse_loss_list, self.wg_loss_list, self.wc_loss_list = pickle.load(file)

    def report(self, mse_loss, wg_loss=None, wc_loss=None):
        self.mse_loss_list.append(mse_loss)
        if wg_loss is not None and wc_loss is not None:
            self.wg_loss_list.append(wg_loss)
            self.wc_loss_list.append(wc_loss)

    def save(self):
        with open(osp.join(self.path, 'loss.bin'), 'wb') as file:
            pickle.dump((self.mse_loss_list, self.wg_loss_list, self.wc_loss_list), file)

    def plot(self, show=False):
        steps = [i for i in range(len(self.mse_loss_list))]
        plt.figure(figsize=(10,5))
        plt.plot(steps, self.mse_loss_list)
        plt.xlabel('Épocas')
        plt.ylabel('Custo MSE')
        plt.tight_layout()
        plt.savefig(osp.join(self.path, "mse_loss.pdf"), bbox_inches='tight')
        if show:
            plt.show()

        if self.wc_loss_list != [] and self.wg_loss_list != []:
            plt.figure(figsize=(10,5))
            plt.plot(self.wg_loss_list,label="G")
            plt.plot(self.wc_loss_list,label="D")
            plt.xlabel("Épocas")
            plt.ylabel("Custo Wasserstein")
            plt.legend()
            plt.tight_layout()
            plt.savefig(osp.join(self.path, "w_loss.pdf"))
            if show:
                plt.show()
        