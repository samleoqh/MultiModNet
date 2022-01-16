import os
import numpy as np
import datetime

from data.agri_vision_2021.loader import *


class agriculture2021_configs(object):
    model = 'none'
    # data set parameters
    dataset = 'Agriculture2021'
    bands = ['RGB', 'NIR']
    loader = AlgricultureDataset
    labels = land_classes
    nb_classes = len(land_classes)

    # model configures
    hidden_ch = 32
    kernel = 8
    heads = 1
    depth = 1
    m_views = 3
    dropout = 0.3
    activation = None
    shortcut = True
    ndvi = True

    palette = palette_vsl  # palette_vsl

    weights = []

    # noise_label = False
    non_val = False
    k_folder = 6
    k = 0
    input_size = [512, 512]
    scale_rate = 1.0/1.0
    val_size = [512, 512]
    train_samples = 56944
    val_samples = 18334
    train_batch = 12
    val_batch = 12

    # flag of mean_std normalization to [-1, 1]
    pre_norm = False
    seeds = 1602 # random seed

    # training hyper parameters
    optim = 'Adam' # 'SGD'
    lr = 2.8e-4
    lr_decay = 0.9
    max_iter = 1e8

    # Cosine schedule parameter
    lrschd = 'Cosin' # else step
    cosin_epoch = 40
    min_lr = 1.18e-7

    # StepLR schedule parameter
    steps = 5
    gamma = 0.78  # lr = lr * (gamma ^ epoch//steps)


    # l2 regularization factor, increasing or decreasing to fight over-fitting
    weight_decay = 2e-5
    momentum = 0.9

    # check point parameters
    ckpt_path = '../ckpt'
    snapshot = ''
    print_freq = 100
    save_pred = True
    save_rate = 0.2
    best_record = {}

    def __init__(self, net_name=model, data=dataset, bands_list=bands, kf=0, k_folder=0, note=''):
        self.model = net_name
        self.dataset = data
        self.bands = bands_list
        self.k = kf
        self.k_folder = k_folder
        self.suffix_note = note

        if note != 'testing':
            check_mkdir(self.ckpt_path)
            check_mkdir(os.path.join(self.ckpt_path, self.model))

        bandstr = '-'.join(self.bands)
        if self.k_folder is not None:
            subfolder = self.dataset + '_' + bandstr + '_kf-' + str(self.k_folder) + '-' + str(self.k)
        else:
            subfolder = self.dataset + '_' + bandstr
        if note != '':
            subfolder += '-'
            subfolder += note
        if note != 'testing':
            check_mkdir(os.path.join(self.ckpt_path, self.model, subfolder))
            self.save_path = os.path.join(self.ckpt_path, self.model, subfolder)

    def get_file_list(self):
        return split_train_val_test_sets(name=self.dataset, bands=self.bands, KF=self.k_folder, k=self.k, seeds=self.seeds, non_val=self.non_val)

    def get_dataset(self):
        train_dict, val_dict, test_dict = self.get_file_list()
        # split_train_val_test_sets(name=self.dataset,KF=None, k=self.k,seeds=self.seeds)
        train_set = self.loader(mode='train', file_lists=train_dict, pre_norm=self.pre_norm,
                                    num_samples=self.train_samples, windSize=self.input_size, scale=self.scale_rate)
        val_set = self.loader(mode='val', file_lists=val_dict, pre_norm=self.pre_norm,
                                  num_samples=self.val_samples, windSize=self.val_size, scale=self.scale_rate)
        return train_set, val_set


    def resume_train(self, net):
        if len(self.snapshot) == 0:
            curr_epoch = 1
            self.best_record = {'epoch': 0, 'val_loss': 0, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0,
                                'f1': 0}
        else:
            print('training resumes from ' + self.snapshot)
            # net.load_state_dict(torch.load(self.snapshot))
            net.load_state_dict(torch.load(os.path.join(self.save_path, self.snapshot)))
            split_snapshot = self.snapshot.split('_')
            curr_epoch = int(split_snapshot[1]) + 1
            self.best_record = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11]),
                                'f1': float(split_snapshot[13])}
        return net, curr_epoch


    def print_best_record(self):
        print(
            '[best_ %d]: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1 %.5f]' % (
                self.best_record['epoch'],
                self.best_record['val_loss'], self.best_record['acc'],
                self.best_record['acc_cls'],
                self.best_record['mean_iu'], self.best_record['fwavacc'], self.best_record['f1']
            ))


    def update_best_record(self, epoch, val_loss,
                           acc, acc_cls, mean_iu,
                           fwavacc, f1):
        print('----------------------------------------------------------------------------------------')
        print('[epoch %d]: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1 %.5f]' % (
            epoch, val_loss, acc, acc_cls, mean_iu, fwavacc, f1))
        self.print_best_record()

        print('----------------------------------------------------------------------------------------')
        if mean_iu > self.best_record['mean_iu'] or f1 > self.best_record['f1']:
            self.best_record['epoch'] = epoch
            self.best_record['val_loss'] = val_loss
            self.best_record['acc'] = acc
            self.best_record['acc_cls'] = acc_cls
            self.best_record['mean_iu'] = mean_iu
            self.best_record['fwavacc'] = fwavacc
            self.best_record['f1'] = f1
            return True
        else:
            return False

    def display(self):
        """printout all configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def write2txt(self):
        file = open(os.path.join(self.save_path,
                                 str(datetime.datetime.now()) + '.txt'), 'w')
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                line = "{:30} {}".format(a, getattr(self, a))
                file.write(line + '\n')


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
