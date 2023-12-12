import copy
import numpy as np
import torch
import time
from tqdm import tqdm


class Solver():
    def __init__(self, net, criterion, optimizer, train_dataset, valid_dataset) -> None:
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset


    def train(self, max_epoch=100, disp_freq=-1, check_points=1):
        avg_train_loss = 0
        avg_val_loss = 0
        self.avg_train_loss_set = []
        self.avg_val_loss_set = []
        train_start = np.inf
        valid_start = np.inf
        if check_points == 1:
            self.all_models = {}

        for epoch in range(max_epoch):
            starttime = time.time()
            batch_train_loss = self.train_one_epoch(max_epoch, disp_freq, epoch)
            batch_val_loss = self.validate()
            avg_train_loss += batch_train_loss
            avg_val_loss += batch_val_loss
            self.avg_train_loss_set.append(batch_train_loss)
            self.avg_val_loss_set.append(batch_val_loss)
            if batch_train_loss < train_start:
                train_start = batch_train_loss
                self.train_best_model = copy.deepcopy(self.net)
            if batch_val_loss < valid_start:
                valid_start = batch_val_loss
                self.valid_best_model = copy.deepcopy(self.net)
            if check_points == 1:
                self.all_models['epoch%d' % epoch] = copy.deepcopy(
                    self.net).cpu().state_dict()
            print('Epoch [{}/{}]\t Average training and validation loss: {:.4E} {:.4E}\tTime: {:.2f}s'.format(
                epoch + 1, max_epoch, batch_train_loss, batch_val_loss, time.time() - starttime))


    def train_one_epoch(self, max_epoch, disp_freq, epoch):
        self.net.train()
        train_loss = 0
        iteration = 0
        for data in self.train_dataset:
            if torch.cuda.is_available():
                for i in range(len(data)):
                    data[i] = data[i].cuda()
            iteration += 1
            self.optimizer.zero_grad()
            y = self.net(data[0 : -1])
            loss = self.criterion(y, data[-1])
            # loss = loss.to(torch.double)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            if iteration % disp_freq == 0 and disp_freq > 0:
                print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}".format(
                    epoch + 1, max_epoch, iteration, len(self.train_dataset),
                    train_loss / iteration))
        return train_loss / len(self.train_dataset)


    def validate(self):
        self.net.eval()
        data = self.valid_dataset.dataset.data
        target = self.valid_dataset.dataset.label
        if torch.cuda.is_available():
            for i in range(len(data)):
                data[i] = data[i].cuda()
            target = target.cuda()
        with torch.no_grad():
            y = self.net(data)
        val_loss = self.criterion(y, target)
        return val_loss.item()


    def train_atten(self, max_epoch=100, disp_freq=-1, check_points=1):
        avg_train_loss = 0
        avg_val_loss = 0
        self.avg_train_loss_set = []
        self.avg_val_loss_set = []
        train_start = np.inf
        valid_start = np.inf
        if check_points == 1:
            self.all_models = {}

        for epoch in range(max_epoch):
            starttime = time.time()
            batch_train_loss = self.train_one_epoch_atten(max_epoch, disp_freq, epoch)
            _, batch_val_loss = self.validate_atten()
            avg_train_loss += batch_train_loss
            avg_val_loss += batch_val_loss
            self.avg_train_loss_set.append(batch_train_loss)
            self.avg_val_loss_set.append(batch_val_loss)
            if batch_train_loss < train_start:
                train_start = batch_train_loss
                self.train_best_model = copy.deepcopy(self.net)
            if batch_val_loss < valid_start:
                valid_start = batch_val_loss
                self.valid_best_model = copy.deepcopy(self.net)
            if check_points == 1:
                self.all_models['epoch%d' % epoch] = copy.deepcopy(
                    self.net).cpu().state_dict()
            print('Epoch [{}/{}]\t Average training and validation loss: {:.4E} {:.4E}\tTime: {:.2f}s'.format(
                epoch + 1, max_epoch, batch_train_loss, batch_val_loss, time.time() - starttime))


    def train_one_epoch_atten(self, max_epoch, disp_freq, epoch):
        self.net.train()
        train_loss = 0
        iteration = 0
        for data, target in self.valid_dataset:
            iteration += 1
            # GPU加速
            self.optimizer.zero_grad()
            target_disloc = torch.zeros(target.size()).cuda()
            target_disloc[:, 1:, :] = target[:, :-1, :]
            y = self.net(data, target_disloc)
            loss = self.criterion(y, target)
            # loss = loss.to(torch.double)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            if iteration % disp_freq == 0 and disp_freq > 0:
                print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}".format(
                    epoch + 1, max_epoch, iteration, len(self.valid_dataset),
                    train_loss / iteration))
        return train_loss / len(self.valid_dataset)


    def validate_atten(self):
        self.net.eval()
        val_loss = 0
        prediction = []
        for data, target in self.valid_dataset:
            # GPU加速
            with torch.no_grad():
                target_disloc = torch.zeros(target.size()).cuda()
                target_disloc[:, 1:, :] = target[:, :-1, :]
                y = self.net(data, target_disloc)
            prediction.append(y.cpu().detach().numpy())
            loss = self.criterion(y, target)
            val_loss += loss.item()
        prediction = torch.tensor(prediction)
        return prediction, val_loss / len(self.valid_dataset)



def test(model, criterion, dataset, batch=0):
    test_loss = 0
    model.eval()
    if batch == 0:
        data = dataset.dataset.data
        target = dataset.dataset.label
        if torch.cuda.is_available():
            for i in range(len(data)):
                data[i] = data[i].cuda()
            target = target.cuda()
        with torch.no_grad():
            y = model(data)
        prediction = y.cpu()
        test_loss = criterion(y, target)
    else:
        data = dataset.dataset.data
        target = dataset.dataset.label
        if torch.cuda.is_available():
            for i in range(len(data)):
                data[i] = data[i].cuda()
            target = target.cuda()
        prediction = torch.zeros_like(target)
        num = int(data[0].shape[0] / batch)
        for i in range(num):
            with torch.no_grad():
                prediction[i * batch : i * batch + batch, ...] = model([da[i * batch : i * batch + batch, ...] for da in data])
        if num * batch < data[0].shape[0]:
            with torch.no_grad():
                prediction[num * batch:, ...] = model([da[num * batch:, ...] for da in data])
        test_loss = criterion(prediction, target)
        prediction = prediction.cpu()
    return prediction, test_loss.item()


def test_atten(model, criterion, dataset, batch=0):
    test_loss = 0
    model.eval()
    prediction = []
    pbar = tqdm(dataset, desc='计算中', ncols=100)
    for data, target in pbar:
        # GPU加速
        y_temp = torch.zeros(target.shape[0], 1, target.shape[2])
        with torch.no_grad():
            for i in range(target.shape[-2]):
                y = model(data, y_temp)
                y_temp = torch.concat([y_temp, y[:, -1:, :]], dim=1)
        prediction.append(y.cpu().detach().numpy())
        loss = criterion(y, target)
        test_loss += loss.item()

    prediction = torch.tensor(prediction)
    return prediction, test_loss / len(dataset)
