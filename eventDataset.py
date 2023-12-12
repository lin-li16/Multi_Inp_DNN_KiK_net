import torch
import torchvision
import numpy as np


class eqkDataset(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
    

class eqkDataset_2inp(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data1, data2, label):
        self.data1 = data1
        self.data2 = data2
        self.label = label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        return self.data1[index], self.data2[index], self.label[index]
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data1)
    

class eqkDataset_3inp(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data1, data2, data3, label):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.label = label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        return self.data1[index], self.data2[index], self.data3[index], self.label[index]
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data1)
    

class eqkDataset_multi_inps(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data, label):
        self.data = [torch.tensor(dd) for dd in data]
        self.label = torch.tensor(label)
        self.num_inps = len(data)
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        result = []
        for value in self.data:
            result.append(value[index])
        result.append(self.label[index])
        return tuple(result)
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.label)


class DataProcesser():
    def __init__(self, inp_data, out_data, event_list=None):
        self.inp_data = inp_data
        self.out_data = out_data
        self.event_list = event_list


    # 划分训练集、验证集和测试集的台站
    def divide_stations(self, valid_ratio=0.1, test_ratio=0.2, valid_stations=None, test_stations=None, train_stations=None):
        if valid_stations is None or test_stations is None:
            self.valid_ratio = valid_ratio
            self.test_ratio = test_ratio

            # 统计所有台站名及其地震记录数量
            nums = []
            stations = []
            for sta, events in self.event_list.items():
                nums.append(len(events))
                stations.append(sta)
            
            # 计算训练集、验证集和测试集的地震记录数量
            self.valid_size = int(valid_ratio * sum(nums))
            self.test_size = int(test_ratio * sum(nums))
            self.train_size = sum(nums) - self.valid_size - self.test_size

            # 按训练集、验证集和测试集地震记录数量随机划分三个数据集的台站
            self.valid_stations = []
            self.test_stations = []
            index = np.arange(len(stations))
            np.random.shuffle(index)
            count = 0
            for i, idx in enumerate(index):
                if count <= self.valid_size:
                    self.valid_stations.append(stations[idx])
                elif count > self.valid_size and count <= self.valid_size + self.test_size:
                    self.test_stations.append(stations[idx])
                else:
                    break
                count += nums[idx]
            for sta in self.valid_stations:
                stations.remove(sta)
            for sta in self.test_stations:
                stations.remove(sta)
            self.train_stations = stations
        else:
            # 统计所有台站名及其地震记录数量
            self.valid_stations = valid_stations
            self.test_stations = test_stations
            self.train_size, self.valid_size, self.test_size = 0, 0, 0
            if train_stations is None:
                self.train_stations = []
                for sta, events in self.event_list.items():
                    if sta in valid_stations:
                        self.valid_size += len(events)
                    elif sta in test_stations:
                        self.test_size += len(events)
                    else:
                        self.train_size += len(events)
                        self.train_stations.append(sta)
                self.valid_ratio = self.valid_size / (self.train_size + self.valid_size + self.test_size)
                self.test_ratio = self.test_size / (self.train_size + self.valid_size + self.test_size)
            else:
                self.train_stations = train_stations
                for sta, events in self.event_list.items():
                    if sta in valid_stations:
                        self.valid_size += len(events)
                    elif sta in test_stations:
                        self.test_size += len(events)
                    elif sta in train_stations:
                        self.train_size += len(events)
                self.valid_ratio = self.valid_size / (self.train_size + self.valid_size + self.test_size)
                self.test_ratio = self.test_size / (self.train_size + self.valid_size + self.test_size)
    

    def get_ev_data(self, set_type='train'):
        if set_type == 'train':
            # 获取训练集的地震列表和数据
            self.train_events, self.train_label = [], []
            for sta in self.train_stations:
                for i, ev in enumerate(self.event_list[sta]):
                    self.train_events.append(ev)
                    self.train_label.append(self.out_data[sta][i, ...].astype(np.float32))
            self.train_label = np.array(self.train_label)

            self.train_data = []
            for num, data in enumerate(self.inp_data):
                train_data_single = []
                for sta in self.train_stations:
                    for i, ev in enumerate(self.event_list[sta]):
                        if num == 0:
                            train_data_single.append(data[sta][i, ...])
                        elif num == 1:
                            train_data_single.append(data[sta])
                        else:
                            train_data_single.append(data[ev])
                self.train_data.append(np.array(train_data_single).astype(np.float32))

            self.train_idx = np.arange(len(self.train_events))
            np.random.shuffle(self.train_idx)
            self.train_label = self.train_label[self.train_idx, ...]
            for i in range(len(self.train_data)):
                self.train_data[i] = self.train_data[i][self.train_idx, ...]
        elif set_type == 'valid':
            # 获取验证集的地震列表和数据
            self.valid_events, self.valid_label = [], []
            for sta in self.valid_stations:
                for i, ev in enumerate(self.event_list[sta]):
                    self.valid_events.append(ev)
                    self.valid_label.append(self.out_data[sta][i, ...])
            self.valid_label = np.array(self.valid_label).astype(np.float32)

            self.valid_data = []
            for num, data in enumerate(self.inp_data):
                valid_data_single = []
                for sta in self.valid_stations:
                    for i, ev in enumerate(self.event_list[sta]):
                        if num == 0:
                            valid_data_single.append(data[sta][i, ...])
                        elif num == 1:
                            valid_data_single.append(data[sta])
                        else:
                            valid_data_single.append(data[ev])
                self.valid_data.append(np.array(valid_data_single).astype(np.float32))

            self.valid_idx = np.arange(len(self.valid_events))
            np.random.shuffle(self.valid_idx)
            self.valid_label = self.valid_label[self.valid_idx, ...]
            for i in range(len(self.valid_data)):
                self.valid_data[i] = self.valid_data[i][self.valid_idx, ...]
        elif set_type == 'test':
            # 获取测试集的地震列表和数据
            self.test_events, self.test_label = [], []
            for sta in self.test_stations:
                for i, ev in enumerate(self.event_list[sta]):
                    self.test_events.append(ev)
                    self.test_label.append(self.out_data[sta][i, ...])
            self.test_label = np.array(self.test_label).astype(np.float32)

            self.test_data = []
            for num, data in enumerate(self.inp_data):
                test_data_single = []
                for sta in self.test_stations:
                    for i, ev in enumerate(self.event_list[sta]):
                        if num == 0:
                            test_data_single.append(data[sta][i, ...])
                        elif num == 1:
                            test_data_single.append(data[sta])
                        else:
                            test_data_single.append(data[ev])
                self.test_data.append(np.array(test_data_single).astype(np.float32))

            self.test_idx = np.arange(len(self.test_events))
            np.random.shuffle(self.test_idx)
            self.test_label = self.test_label[self.test_idx, ...]
            for i in range(len(self.test_data)):
                self.test_data[i] = self.test_data[i][self.test_idx, ...]


    def set_logscale(self):
        self.train_data[0] = np.log10(self.train_data[0])
        self.valid_data[0] = np.log10(self.valid_data[0])
        self.test_data[0] = np.log10(self.test_data[0])
        self.train_label = np.log10(self.train_label)
        self.valid_label = np.log10(self.valid_label)
        self.test_label = np.log10(self.test_label)


    def normalize_data(self, normalize='standard'):
        if normalize == 'minmax':
            print('Use the maximum data to normalize the data')
            self.data_max, self.data_min = [], []
            for data in self.train_data:
                self.data_max.append(np.max(data, axis=0))
                self.data_min.append(np.min(data, axis=0))
            self.label_max = np.max(self.train_label, axis=0)
            self.label_min = np.min(self.train_label, axis=0)

            for i in range(len(self.train_data)):
                self.train_data[i] = (self.train_data[i] - self.data_min[i]) / (self.data_max[i] - self.data_min[i])
                self.valid_data[i] = (self.valid_data[i] - self.data_min[i]) / (self.data_max[i] - self.data_min[i])
                self.test_data[i] = (self.test_data[i] - self.data_min[i]) / (self.data_max[i] - self.data_min[i])

            self.train_label = (self.train_label - self.label_min) / (self.label_max - self.label_min)
            self.valid_label = (self.valid_label - self.label_min) / (self.label_max - self.label_min)
            self.test_label = (self.test_label - self.label_min) / (self.label_max - self.label_min)
        elif normalize == 'standard':
            print('Use the mean and std to standard the data')
            self.data_mean, self.data_std = [], []
            for data in self.train_data:
                self.data_mean.append(np.mean(data, axis=0))
                self.data_std.append(np.std(data, axis=0))
            self.label_mean = np.mean(self.train_label, axis=0)
            self.label_std = np.std(self.train_label, axis=0)

            for i in range(len(self.train_data)):
                self.train_data[i] = (self.train_data[i] - self.data_mean[i]) / self.data_std[i]
                self.valid_data[i] = (self.valid_data[i] - self.data_mean[i]) / self.data_std[i]
                self.test_data[i] = (self.test_data[i] - self.data_mean[i]) / self.data_std[i]

            self.train_label = (self.train_label - self.label_mean) / self.label_std
            self.valid_label = (self.valid_label - self.label_mean) / self.label_std
            self.test_label = (self.test_label - self.label_mean) / self.label_std
        

    def make_dataset(self, batch, validratio=0.1, testratio=0.2, normalize=None, valid_stations=None, test_stations=None, train_stations=None, data_scale='normal'):
        # 划分台站
        self.divide_stations(validratio, testratio, valid_stations, test_stations, train_stations)

        # 获取各数据集的地震列表和数据
        self.get_ev_data(set_type='train')
        self.get_ev_data(set_type='valid')
        self.get_ev_data(set_type='test')
        
        # 对输入数据取对数
        self.data_scale = data_scale
        if data_scale == 'log':
            self.set_logscale()

        # 对数据进行标准化
        self.normalize_data(normalize=normalize)

        self.train_dataset = eqkDataset_multi_inps(self.train_data, self.train_label)
        self.valid_dataset = eqkDataset_multi_inps(self.valid_data, self.valid_label)
        self.test_dataset = eqkDataset_multi_inps(self.test_data, self.test_label)

        self.batch = batch
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset)


    def make_dataset_events(self, batch, validratio=0.1, testratio=0.2, normalize=None, data_scale='normal'):
        # 获取各数据集的地震列表和数据
        self.train_events, self.train_label = [], []
        self.valid_events, self.valid_label = [], []
        self.test_events, self.test_label = [], []
        self.train_size, self.valid_size, self.test_size = 0, 0, 0
        for sta in list(self.event_list.keys()):
            valid_size = validratio * len(self.event_list[sta])
            test_size = testratio * len(self.event_list[sta])
            train_size = len(self.event_list[sta]) - valid_size - test_size
            self.train_size += train_size
            self.valid_size += valid_size
            self.test_size += test_size
            for i, ev in enumerate(self.event_list[sta]):
                if i < train_size:
                    self.train_events.append(ev)
                    self.train_label.append(self.out_data[sta][i, ...].astype(np.float32))
                elif i >= train_size and i < train_size + valid_size:
                    self.valid_events.append(ev)
                    self.valid_label.append(self.out_data[sta][i, ...].astype(np.float32))
                else:
                    self.test_events.append(ev)
                    self.test_label.append(self.out_data[sta][i, ...].astype(np.float32))
        self.train_label = np.array(self.train_label)
        self.valid_label = np.array(self.valid_label)
        self.test_label = np.array(self.test_label)

        self.train_data, self.valid_data, self.test_data = [], [], []
        for num, data in enumerate(self.inp_data):
            train_data_single, valid_data_single, test_data_single = [], [], []
            for sta in list(self.event_list.keys()):
                valid_size = validratio * len(self.event_list[sta])
                test_size = testratio * len(self.event_list[sta])
                train_size = len(self.event_list[sta]) - valid_size - test_size
                for i, ev in enumerate(self.event_list[sta]):
                    if i < train_size:
                        if num == 0:
                            train_data_single.append(data[sta][i, ...])
                        elif num == 1:
                            train_data_single.append(data[sta])
                        else:
                            train_data_single.append(data[ev])
                    elif i >= train_size and i < train_size + valid_size:
                        if num == 0:
                            valid_data_single.append(data[sta][i, ...])
                        elif num == 1:
                            valid_data_single.append(data[sta])
                        else:
                            valid_data_single.append(data[ev])
                    else:
                        if num == 0:
                            test_data_single.append(data[sta][i, ...])
                        elif num == 1:
                            test_data_single.append(data[sta])
                        else:
                            test_data_single.append(data[ev])
            self.train_data.append(np.array(train_data_single).astype(np.float32))
            self.valid_data.append(np.array(valid_data_single).astype(np.float32))
            self.test_data.append(np.array(test_data_single).astype(np.float32))

        self.train_idx = np.arange(len(self.train_events))
        self.valid_idx = np.arange(len(self.valid_events))
        self.test_idx = np.arange(len(self.test_events))
        
        # 对输入数据取对数
        self.data_scale = data_scale
        if data_scale == 'log':
            self.set_logscale()

        # 对数据进行标准化
        self.normalize_data(normalize=normalize)

        self.train_dataset = eqkDataset_multi_inps(self.train_data, self.train_label)
        self.valid_dataset = eqkDataset_multi_inps(self.valid_data, self.valid_label)
        self.test_dataset = eqkDataset_multi_inps(self.test_data, self.test_label)

        self.batch = batch
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset)