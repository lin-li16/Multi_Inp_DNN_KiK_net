# 导入必要的库
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import warnings
import time
import os
import sys
import scipy.io
import pickle
from tqdm import tqdm
from net import *
from solver import Solver, test
from plot import plot_loss
from eventDataset import *
from get_performance import *
from torchinfo import summary
warnings.filterwarnings("ignore")
sns.set_style('ticks')
sns.set_context("poster")
plt.rcParams['font.sans-serif'] = 'Times New Roman'


class Logger(object):
    '''
    log文件记录对象，将所有print信息记录在log文件中
    '''
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass


def main():
    # 添加命令行输入参数
    parser = argparse.ArgumentParser(description='DNN Model for Time Series Forecasting in KiK-Net Downhole Array Dataset')
    parser.add_argument('--path', type=str, default='attn_2inp_model', help='Parent file path of the dataset and the results')
    parser.add_argument('--batch', type=int, default=1024, help='Batch size of training data')
    parser.add_argument('--validratio', type=float, default=0.1, help='Ratio of validation data in all data, 0-1.0')
    parser.add_argument('--testratio', type=float, default=0.2, help='Ratio of test data in all data, 0-1.0')
    parser.add_argument('--fixedorder', type=int, default=0, help='Whether to use the former data order')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--printfreq', type=int, default=-1, help='Training message print frequency in each epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, default='Attn_2inp', help='Type of model used in this dataset')
    parser.add_argument('--numlayers', type=int, default=3, help='Number of layers in LSTM')
    parser.add_argument('--resultspath', type=str, default='attn_3inps_100class_sa0', help='File path of results')
    parser.add_argument('--kernel', type=int, default=5, help='Kernel size used in CNN layers')
    parser.add_argument('--normalize', type=str, default='standard', help='Normalization of the dataset')
    parser.add_argument('--pretrain', type=str, default='no', help='Use the pre-trained model')
    parser.add_argument('--checkpoints', type=int, default=0, help='Output the models at all epochs')
    parser.add_argument('--plots', type=int, default=1, help='Plot the figures for each earthquake')
    parser.add_argument('--noisy', type=float, default=0, help='Noisy level added to the data')
    parser.add_argument('--bias', type=int, default=1, help='Bias in network')
    parser.add_argument('--datapre', type=str, default='data_preprocess_100class_logsa.pkl', help='DataProcesser file')
    args = parser.parse_args()

    # 创建结果文件夹
    results_path = os.path.join(args.path, args.resultspath)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        os.mkdir(os.path.join(results_path, 'figures'))
    sys.stdout = Logger(os.path.join(results_path, 'message.log'))      # 创建log文件对象
    print('The path of the results is %s' % results_path)

    ## 导入数据
    model_path = args.path
    print('Load data from %s' % model_path)
    if args.datapre is not None:
        with open(os.path.join(model_path, args.datapre), 'rb') as file:
            data_set = pickle.loads(file.read())
    else:
        if model_path == 'inp_arg3_model':
            event_list = np.load(os.path.join(model_path, 'event.npy'), allow_pickle=True).item()
            vsvp = np.load(os.path.join(model_path, 'vsvp.npy'), allow_pickle=True).item()
            source = np.load(os.path.join(model_path, 'source.npy'), allow_pickle=True).item()
            sa_dh = np.load(os.path.join(model_path, 'Sa_dh_stations.npy'), allow_pickle=True).item()
            sa_up = np.load(os.path.join(model_path, 'Sa_up_stations.npy'), allow_pickle=True).item()
            data_set = DataProcesser([sa_dh, vsvp, source], sa_up, event_list)
        elif model_path == 'inp_arg2_model':
            event_list = np.load(os.path.join(model_path, 'event.npy'), allow_pickle=True).item()
            vsvp = np.load(os.path.join(model_path, 'vsvp.npy'), allow_pickle=True).item()
            sa_dh = np.load(os.path.join(model_path, 'Sa_dh_stations.npy'), allow_pickle=True).item()
            sa_up = np.load(os.path.join(model_path, 'Sa_up_stations.npy'), allow_pickle=True).item()
            data_set = DataProcesser([sa_dh, vsvp], sa_up, event_list)
        elif model_path == 'inp_arg1_model':
            event_list = np.load(os.path.join(model_path, 'event.npy'), allow_pickle=True).item()
            sa_dh = np.load(os.path.join(model_path, 'Sa_dh_stations.npy'), allow_pickle=True).item()
            sa_up = np.load(os.path.join(model_path, 'Sa_up_stations.npy'), allow_pickle=True).item()
            data_set = DataProcesser([sa_dh], sa_up, event_list)
    
        ## 构造训练集、验证集和测试集
        data_set.make_dataset(args.batch, args.validratio, args.testratio, args.normalize)
        data_set_out = open(os.path.join(model_path, 'data_preprocess.pkl'), 'wb')
        data_set_out.write(pickle.dumps(data_set))
        data_set_out.close()

    # 建立模型
    max_epoch = args.epochs
    disp_freq = args.printfreq
    learning_rate = args.lr
    print('%s model is applied' % args.model)
    print('Learning rate is %f' % learning_rate)
    if args.model == 'Basic_3inp':
        Net = Basic_3inp(kernel_size=args.kernel)
    elif args.model == 'Basic_2inp':
        Net = Basic_2inp(kernel_size=args.kernel)
    elif args.model == 'Basic_1inp':
        Net = Basic_1inp(kernel_size=args.kernel)
    elif args.model == 'Attn_2inp':
        Net = Attn_2inp()
    summary(Net)
    
    if args.pretrain != 'no':
        print('Apply pretrain model!')
        model_pre = torch.load(os.path.join(args.pretrain, 'results_' + args.model, 'validbest.pt'))
        Net.load_state_dict(model_pre.state_dict())
    # GPU加速
    if torch.cuda.is_available():
        Net = Net.cuda()
    # optimizer = torch.optim.LBFGS(Net.parameters(), lr=learning_rate, max_iter=2)
    optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    slvr = Solver(Net, criterion, optimizer, data_set.train_loader, data_set.valid_loader)
    starttime = time.time()
    slvr.train(max_epoch, disp_freq, check_points=args.checkpoints)
    train_time = time.time()-starttime
    print('Training Time {:.4f}'.format(train_time))
    _, test_loss = test(slvr.valid_best_model, criterion, data_set.test_loader)
    print("Test Loss {:.4f}\n".format(test_loss))
    torch.cuda.empty_cache()

    # 绘制loss变化曲线
    plot_loss(slvr.avg_train_loss_set, slvr.avg_val_loss_set, yscale='log')
    plt.savefig(os.path.join(results_path, 'loss.svg'), bbox_inches='tight')
    print('Training best epoch: %d\tTraining minimum loss: %.3E' % (np.argmin(slvr.avg_train_loss_set) + 1, np.min(slvr.avg_train_loss_set)))
    print('Validate best epoch: %d\tValidate minimum loss: %.3E' % (np.argmin(slvr.avg_val_loss_set) + 1, np.min(slvr.avg_val_loss_set)))
    torch.save(slvr.train_best_model, os.path.join(results_path, 'trainbest.pt'))
    torch.save(slvr.valid_best_model, os.path.join(results_path, 'validbest.pt'))
    if args.checkpoints==0:
        torch.save(slvr.net, os.path.join(results_path, 'last.pt'))
    else:
        torch.save(slvr.all_models, os.path.join(results_path, 'allmodels.pt'))    

    # 结果处理
    Period = np.logspace(np.log10(0.01), np.log10(10), 200)
    postprocess = PostProcess(slvr.valid_best_model, data_set, x_out=Period)
    postprocess.getResults(args, results_path, train_time, slvr.avg_train_loss_set, slvr.avg_val_loss_set)
    if args.plots == 1:
        postprocess.plotResults(results_path, 'Period (s)', 'Sa (g)', dim=0, scale='logx')


if __name__ == "__main__":
    main()