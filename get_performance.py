import numpy as np
import torch
import torch.nn as nn
# import matlab.engine
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
import os
from tqdm import tqdm
from solver import test
# matlabeng = matlab.engine.start_matlab()   #启动matlab
sns.set_style('ticks')
sns.set_context("poster")
plt.rcParams['font.sans-serif'] = 'Arial'
matplotlib.use('AGG')


class PostProcess():
    def __init__(self, model, dataset, x_inp=None, x_out=None) -> None:
        self.model = model
        self.dataset = dataset
        self.x_inp = x_inp
        self.x_out = x_out


    def __performance(self, label, pred, type='mean'):
        if type == 'mean':
            if len(label.shape) == 3:
                MSE = np.sqrt(np.mean((pred - label) ** 2))
                label_std = np.sqrt(np.mean((label - np.mean(label, axis=1)[:, None, :]) ** 2, axis=1))
                RMSE = np.mean(np.sqrt(np.mean((pred - label) ** 2, axis=1)) / label_std)
                MAE = np.mean(np.abs(pred - label))
                RMAE = np.mean(np.max(np.abs(pred - label), axis=1) / label_std)
                r_all = np.zeros(label.shape[0])
                for i in range(label.shape[0]):
                    r_all[i] = np.corrcoef(label[i, :, :].ravel(), pred[i, :, :].ravel())[0, 1]
                r = np.mean(r_all)
            elif len(label.shape) == 2:
                MSE = np.sqrt(np.mean(pred - label) ** 2)
                label_std = np.sqrt(np.mean((label - np.mean(label, axis=0)[None, :]) ** 2), axis=0)
                RMSE = np.mean(np.sqrt(np.mean((pred - label) ** 2, axis=0)) / label_std)
                MAE = np.mean(np.abs(pred - label))
                RMAE = np.mean(np.max(np.abs(pred - label), axis=0) / label_std)
                r = np.corrcoef(label.ravel(), pred.ravel())[0, 1]
            performance = {'MSE': MSE, 'RMSE': RMSE, 'MAE': MAE, 'RMAE': RMAE, 'r': r}
        else:
            MSE = np.sqrt(np.mean((pred - label) ** 2, axis=1))
            label_std = np.sqrt(np.mean((label - np.mean(label, axis=1)[:, None, :]) ** 2, axis=1))
            RMSE = MSE / label_std
            MAE = np.mean(np.abs(pred - label), axis=1)
            RMAE = np.max(np.abs(pred - label), axis=1) / label_std
            r = np.zeros(label.shape[0])
            for i in range(label.shape[0]):
                r[i] = np.corrcoef(label[i, :, :].ravel(), pred[i, :, :].ravel())[0, 1]
            performance = {'MSE': MSE, 'RMSE': RMSE, 'MAE': MAE, 'RMAE': RMAE, 'r': r}
        return performance


    def __pred(self, args, data_type):
        if data_type == 'train':
            data, label = self.dataset.train_data, self.dataset.train_label
            pred, _ = test(self.model, nn.MSELoss(), torch.utils.data.DataLoader(self.dataset.train_dataset), self.dataset.batch)
        elif data_type == 'valid':
            data, label = self.dataset.valid_data, self.dataset.valid_label
            pred, _ = test(self.model, nn.MSELoss(), torch.utils.data.DataLoader(self.dataset.valid_dataset))
        else:
            data, label = self.dataset.test_data, self.dataset.test_label
            pred, _ = test(self.model, nn.MSELoss(), torch.utils.data.DataLoader(self.dataset.test_dataset))
        pred = np.array(pred)
 
        if args.normalize == 'minmax':
            for i in range(len(data)):
                data[i] = data[i] * (self.dataset.data_max[i] - self.dataset.data_min[i]) + self.dataset.data_min[i]
            label = label * (self.dataset.label_max - self.dataset.label_min) + self.dataset.label_min
            pred = pred * (self.dataset.label_max - self.dataset.label_min) + self.dataset.label_min
        elif args.normalize == 'standard':
            for i in range(len(data)):
                data[i] = data[i] * self.dataset.data_std[i] + self.dataset.data_mean[i]
            label = label * self.dataset.label_std + self.dataset.label_mean
            pred = pred * self.dataset.label_std + self.dataset.label_mean

        if args.noisy > 0:
            label = self.dataset.out_data[self.dataset.train_idx, :, :]

        return data, label, pred


    def getResults(self, args, results_path, train_time, train_loss, valid_loss, recover=None):
        self.dataset.train_data, self.dataset.train_label, self.dataset.train_pred = self.__pred(args, 'train')
        if recover is not None:
            self.dataset.train_data, self.dataset.train_label, self.dataset.train_pred = self.dataset.recoverData(self.dataset.train_data, self.dataset.train_label, self.dataset.train_pred, recover)
        self.train_performance = self.__performance(self.dataset.train_label, self.dataset.train_pred)
        print('Train set | ', end='')
        for key, value in self.train_performance.items():
            print('%s: %.3E, ' % (key, value), end='')
        print('\n')

        self.dataset.valid_data, self.dataset.valid_label, self.dataset.valid_pred = self.__pred(args, 'valid')
        if recover is not None:
            self.dataset.valid_data, self.dataset.valid_label, self.dataset.valid_pred = self.dataset.recoverData(self.dataset.valid_data, self.dataset.valid_label, self.dataset.valid_pred, recover)
        self.valid_performance = self.__performance(self.dataset.valid_label, self.dataset.valid_pred)
        print('Valid set | ', end='')
        for key, value in self.valid_performance.items():
            print('%s: %.3E, ' % (key, value), end='')
        print('\n')

        self.dataset.test_data, self.dataset.test_label, self.dataset.test_pred = self.__pred(args, 'test')
        if recover is not None:
            self.dataset.test_data, self.dataset.test_label, self.dataset.test_pred = self.dataset.recoverData(self.dataset.test_data, self.dataset.test_label, self.dataset.test_pred, recover)
        self.test_performance = self.__performance(self.dataset.test_label, self.dataset.test_pred)
        print('Test set | ', end='')
        for key, value in self.test_performance.items():
            print('%s: %.3E, ' % (key, value), end='')
        print('\n')

        ## 输出performance数据
        perfile = open(os.path.join(results_path, 'performance.out'), 'w')
        datatype = ['Train', 'Valid', 'Test']
        allperformance = [self.train_performance, self.valid_performance, self.test_performance]
        perfile.write('训练总次数:\t\t%d\n' % args.epochs)
        perfile.write('训练总时间:\t\t%.2f\n' % train_time)
        perfile.write('训练最好次数:\t\t%d\n' % np.argmin(train_loss))
        perfile.write('验证最好次数:\t\t%d\n' % np.argmin(valid_loss))
        for i, per in enumerate(allperformance):
            for key, value in per.items():
                if key == 'r':
                    perfile.write(datatype[i] + '-' + key + ':\t\t%.1f\n' % (100 * value))
                else:
                    perfile.write(datatype[i] + '-' + key + ':\t\t%.3E\n' % value)
        perfile.close()

        # 保存结果数据
        scipy.io.savemat(os.path.join(results_path, 'result.mat'),
                        {'train_pred': self.dataset.train_pred, 'test_pred': self.dataset.test_pred, 'valid_pred': self.dataset.valid_pred, 'train_loss': train_loss, 'valid_loss': valid_loss, 'train_performance': self.train_performance, 'valid_performance': self.valid_performance, 'test_performance': self.test_performance})


    def __plotOneSet(self, results_path, label, pred, legend, xlabel, ylabel, dim=0, lgd_each=None, scale='linear'):
        performance = self.__performance(label, pred, type='all')
        if label.shape[0] > 200:
            pbar = tqdm(range(0, label.shape[0], int(label.shape[0] / 200)), desc=legend + ' Plotting', ncols=100)
        else:
            pbar = tqdm(range(0, label.shape[0]), desc=legend + ' Plotting', ncols=100)
        for i in pbar:
            plt.figure(figsize=(8, 6))
            if self.x_out is not None:
                if scale == 'linear':
                    plt.plot(self.x_out, label[i, :, dim], 'k', label='Label')
                    plt.plot(self.x_out, pred[i, :, dim], 'r--', label='Prediction')
                elif scale == 'logx':
                    plt.semilogx(self.x_out, label[i, :, dim], 'k', label='Label')
                    plt.semilogx(self.x_out, pred[i, :, dim], 'r--', label='Prediction')
                elif scale == 'logy':
                    plt.semilogy(self.x_out, label[i, :, dim], 'k', label='Label')
                    plt.semilogy(self.x_out, pred[i, :, dim], 'r--', label='Prediction')
                elif scale == 'loglog':
                    plt.loglog(self.x_out, label[i, :, dim], 'k', label='Label')
                    plt.loglog(self.x_out, pred[i, :, dim], 'r--', label='Prediction')
                else:
                    print('Plot scale error!')
            else:
                plt.plot(label[i, :, dim], 'k', label='Label')
                plt.plot(pred[i, :, dim], 'r--', label='Prediction')
            plt.legend()
            # plt.xlim([0.3, 20])
            # plt.ylim([0.5, 30])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if lgd_each is None:
                plt.savefig(os.path.join(results_path, 'figures', '%s%d_e%.3f_r%.1f.svg' % (legend, i, performance['MSE'][i], 100 * performance['r'][i])), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(results_path, 'figures', '%s%d_%s_e%.3f_r%.1f.svg' % (legend, i, lgd_each[i], performance['MSE'][i], 100 * performance['r'][i])), bbox_inches='tight')
            plt.close('all')
        return performance


    def plotResults(self, results_path, xlabel, ylabel, dim=0, lgd_each=None, scale='linear'):
        train_per = self.__plotOneSet(results_path, self.dataset.train_label, self.dataset.train_pred, 'Train', xlabel, ylabel, dim, lgd_each, scale)
        valid_per = self.__plotOneSet(results_path, self.dataset.valid_label, self.dataset.valid_pred, 'Valid', xlabel, ylabel, dim, lgd_each, scale)
        test_per = self.__plotOneSet(results_path, self.dataset.test_label, self.dataset.test_pred, 'Test', xlabel, ylabel, dim, lgd_each, scale)
        scipy.io.savemat(os.path.join(results_path, 'performance.mat'),
                        {'train-performance': train_per, 'valid-performance': valid_per, 'test-performance': test_per})