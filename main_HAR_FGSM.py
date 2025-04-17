import pandas as pd

from data_loader_HAR import data_generator, count_parameters

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os

import Model
from sklearn.metrics import precision_recall_fscore_support

import argparse
import matplotlib.pyplot as plt
import random
import csv
import uuid
import tqdm


def create_csv_file(list1, list2, list3, filename):
    rows = zip(list1, list2, list3)
    file_path = filename

    with open(file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['EPOCH', 'TEST_ACCURACY', 'AVG_MAF1_SCORE'])
        csv_writer.writerows(rows)


class Train():
    def __init__(self, args):

        self.train, self.valid, self.test = data_generator('Tianyu_HAR_Processed', args=args)

        self.args = args
        self.net = Model.FC_STGNN_HAR(args.patch_size, args.conv_out, args.lstmhidden_dim, args.lstmout_dim,
                                      args.conv_kernel, args.hidden_dim, args.time_denpen_len, args.num_sensor,
                                      args.num_windows, args.moving_window, args.stride, args.decay, args.pool_choice,
                                      args.n_class)

        self.net = self.net.cuda() if tr.cuda.is_available() else self.net
        self.loss_function = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.net.parameters())

        print("# of parameters:", count_parameters(self.net))

    def Train_batch(self):
        self.net.train()
        loss_ = 0
        for data, label in self.train:
            data = data.cuda() if tr.cuda.is_available() else data
            label = label.cuda() if tr.cuda.is_available() else label
            self.optim.zero_grad()
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()
        return loss_

    def Train_model(self):
        epoch = self.args.epoch
        cross_accu = 0
        test_accu_ = []
        prediction_ = []
        real_ = []
        Epoches = []
        maf1_scores = []

        for i in range(epoch):
            time0 = time.time()
            loss = self.Train_batch()
            if i % self.args.show_interval == 0:
                accu_val = self.Cross_validation()

                if accu_val > cross_accu:
                    cross_accu = accu_val
                    test_accu, prediction, real, maf1_score = self.Prediction()

                    print('>>>>>>>>>>>>> {}th epoch, TRAINING accuracy is {}%, TESTING accuracy is {}%, maf1_score is {}%'.format(i, 
                          np.round(accu_val, 3), np.round(test_accu, 3), np.round(maf1_score, 3)))

                    maf1_scores.append(maf1_score)
                    test_accu_.append(test_accu)
                    Epoches.append(i)
                    # why is this commented?
                    prediction_.append(prediction.detach().numpy())
                    real_.append(real.detach().numpy())
                    self.save()

        # args.save_name = "/home/xaviar/FC-STGNN/HAR_DATASET/UCI HAR Dataset_pre_processed/TRAIN_3.csv"
        # create_csv_file(Epoches, test_accu_, maf1_scores, args.save_name)
        np.save(f'experiment/HAR_{uuid.uuid4()}.npy', np.array([test_accu_, maf1_scores, prediction_, real_], dtype=object))

    def cuda_(self, x):
        x = tr.Tensor(np.array(x))

        if tr.cuda.is_available():
            return x.cuda()
        else:
            return x

    def Cross_validation(self):
        self.net.eval()
        prediction_ = []
        real_ = []

        for data, label in self.valid:
            data = data.cuda() if tr.cuda.is_available() else data
            real_.append(label)
            prediction = self.net(data)
            prediction_.append(prediction.detach().cpu())
        prediction_ = tr.cat(prediction_, 0)
        real_ = tr.cat(real_, 0)

        prediction_ = tr.argmax(prediction_, -1)
        accu = self.accu_(prediction_, real_)
        # print(accu)
        return accu

    def Prediction(self):
        self.net.eval()
        prediction_ = []
        real_ = []
        for data, label in self.test:
            data = data.cuda() if tr.cuda.is_available() else data
            real_.append(label)
            prediction = self.net(data)
            prediction_.append(prediction.detach().cpu())
        prediction_ = tr.cat(prediction_, 0)
        real_ = tr.cat(real_, 0)

        prediction_ = tr.argmax(prediction_, -1)
        accu = self.accu_(prediction_, real_)
        maf1 = self.maf1(predicted=prediction_, real=real_)
        return accu, prediction_, real_, maf1

    def accu_(self, predicted, real):
        num = predicted.size(0)
        real_num = 0
        for i in range(num):
            if predicted[i] == real[i]:
                real_num += 1
        return 100 * real_num / num

    def maf1(self, predicted, real):
        _, _, f1, _ = precision_recall_fscore_support(real, predicted, average="macro")
        return 100 * f1
    
    def save(self):
        tr.save(self.net.state_dict(), "HAR_PAMAP_trained")
    
    def load(self):
        self.net.load_state_dict(tr.load("HAR_PAMAP_trained"))
    
    def pertubate(self):
        # self.net.train()
        self.net.eval()
        accus, mafs = list(), list()
        norm = 0
        num_features = 20
        for epsilon in tqdm.tqdm(np.linspace(0, 0.1, 15)):
            prediction_ = []
            real_ = []
            for data, label in self.test:
                real_.append(label)
                if epsilon != 0:
                    data = data.cuda() if tr.cuda.is_available() else data
                    label = label.cuda() if tr.cuda.is_available() else label
                    data.requires_grad = True
                    
                    prediction = self.net(data)
                    loss = self.loss_function(prediction, label)
                    loss.backward()
                    
                    if norm == np.inf:
                        pertubation = epsilon * 2 * data.grad.data.sign()
                    elif norm != 0:
                        gradient = data.grad
                        pertubation = tr.zeros(gradient.shape)
                        pertubation = pertubation.cuda() if tr.cuda.is_available() else pertubation

                        for i in range(gradient.shape[0]):
                            denominator = np.linalg.norm(gradient[i].data.cpu().numpy().flatten(), norm)

                            norm_grad = gradient[i].data / denominator
                            pertubation[i] = norm_grad * epsilon * 2
                    else:
                        gradient = data.grad
                        pertubation = tr.zeros(gradient.shape)
                        pertubation = pertubation.cuda() if tr.cuda.is_available() else pertubation

                        for i in range(gradient.shape[0]):
                            grads = gradient[i].data.cpu().numpy()
                            indexes = np.argpartition(grads.reshape(-1), -num_features)[-num_features:]
                            pertubation[i].reshape(-1)[indexes] = gradient[i].data.reshape(-1)[indexes].sign() * epsilon * 2

                    x_adv = tr.clamp(data + pertubation, -1, 1)
                else:
                    x_adv = data.cuda() if tr.cuda.is_available() else data

                prediction = self.net(x_adv)
                prediction_.append(prediction.detach().cpu())
            
            prediction_ = tr.cat(prediction_, 0)
            real_ = tr.cat(real_, 0)

            prediction_ = tr.argmax(prediction_, -1)
            accu = self.accu_(prediction_, real_)
            maf1 = self.maf1(predicted=prediction_, real=real_)
            accus.append(accu)
            mafs.append(maf1)

        print(accus, mafs)


if __name__ == '__main__':
    from args import args

    args = args()


    def args_config_HAR(args):
        args.epoch = 50
        args.k = 1
        args.window_sample = 128

        args.decay = 0.7
        args.pool_choice = 'mean'
        args.moving_window = [2, 2]
        args.stride = [1, 2]
        args.lr = 1e-3
        args.batch_size = 100

        args.conv_kernel = 6
        args.patch_size = 64
        args.time_denpen_len = int(args.window_sample / args.patch_size)
        args.conv_out = 10
        args.num_windows = 2

        args.conv_time_CNN = 6

        args.lstmout_dim = 18
        args.hidden_dim = 16
        args.lstmhidden_dim = 48

        args.num_sensor = 9
        args.n_class = 6
        return args


    args = args_config_HAR(args)

    train = Train(args)
    # train.Train_model()
    train.load()
    train.pertubate()
