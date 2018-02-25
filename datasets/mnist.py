
import torch as T
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from itertools import product
from util import *
import os
import cv2
import numpy as NP
import numpy.random as RNG

def mnist_bbox(data):
    n_rows, n_cols = data.size()
    rowwise_max = data.max(0)[0]
    colwise_max = data.max(1)[0]
    rowwise_max_mask = rowwise_max == 0
    colwise_max_mask = colwise_max == 0

    left = T.cumprod(rowwise_max_mask, 0).sum()
    top = T.cumprod(colwise_max_mask, 0).sum()
    right = n_cols - T.cumprod(reverse(rowwise_max_mask, 0), 0).sum()
    bottom = n_rows - T.cumprod(reverse(colwise_max_mask, 0), 0).sum()

    x = (left + right) / 2
    y = (top + bottom) / 2
    w = right - left
    h = bottom - top

    return T.FloatTensor([x, y, w, h])

class MNISTMulti(Dataset):
    dir_ = 'multi'

    @property
    def _meta(self):
        return '%d-%d-%d-%d.pt' % (
                self.image_rows,
                self.image_cols,
                self.n_digits,
                self.backrand)

    @property
    def training_file(self):
        return os.path.join(self.dir_, 'training-' + self._meta)

    @property
    def test_file(self):
        return os.path.join(self.dir_, 'test-' + self._meta)

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 image_rows=100,
                 image_cols=100,
                 n_digits=1,
                 backrand=0):
        self.train = train
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.n_digits = n_digits
        self.backrand = backrand

        if os.path.exists(self.dir_):
            if os.path.isfile(self.dir_):
                raise NotADirectoryError(self.dir_)
            elif os.path.exists(self.training_file) and train:
                data = T.load(self.training_file)
                self.train_data = data['data']
                self.train_labels = data['labels']
                self.train_locs = data['locs']
                self.size = self.train_data.size()[0]
                return
            elif os.path.exists(self.test_file) and not train:
                data = T.load(self.test_file)
                self.test_data = data['data']
                self.test_labels = data['labels']
                self.test_locs = data['locs']
                self.size = self.test_data.size()[0]
                return
        elif not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

        for _train in [False, True]:
            mnist = MNIST(root, _train, transform, target_transform, download)
            src_data = getattr(mnist, 'train_data' if _train else 'test_data')
            src_labels = getattr(mnist, 'train_labels' if _train else 'test_labels')

            with T.random.fork_rng():
                T.random.manual_seed(1000 if _train else 2000)

                n_samples, n_rows, n_cols = src_data.size()
                n_new_samples = n_samples * n_digits
                data = T.ByteTensor(n_new_samples, image_rows, image_cols).zero_()
                labels = T.LongTensor(n_new_samples, n_digits).zero_()
                locs = T.LongTensor(n_new_samples, n_digits, 4).zero_()

                for i, j in product(range(n_digits), range(n_digits)):
                    pos_rows = (T.LongTensor(n_samples).random_() %
                                (image_rows - n_rows))
                    pos_cols = (T.LongTensor(n_samples).random_() %
                                (image_cols - n_cols))
                    perm = T.randperm(n_samples)
                    for k, idx in zip(
                            range(n_samples * j, n_samples * (j + 1)), perm):
                        cur_rows = RNG.randint(n_rows // 3 * 2, n_rows)
                        cur_cols = RNG.randint(n_rows // 3 * 2, n_cols)
                        row = RNG.randint(image_rows - cur_rows)
                        col = RNG.randint(image_cols - cur_cols)
                        cur_data = T.from_numpy(
                                cv2.resize(
                                    src_data[idx].numpy(),
                                    (cur_cols, cur_rows))
                                )
                        data[k, row:row+cur_rows, col:col+cur_cols][cur_data != 0] = cur_data[cur_data != 0]
                        labels[k, i] = src_labels[idx]
                        locs[k, i] = mnist_bbox(cur_data)
                        locs[k, i, 0] += col
                        locs[k, i, 1] += row

                if backrand:
                    data += (data.new(*data.size()).random_() % backrand) * (data == 0)

            T.save({
                'data': data,
                'labels': labels,
                'locs': locs,
                }, self.training_file if _train else self.test_file)

            if train and _train:
                self.train_data = data
                self.train_labels = labels
                self.train_locs = locs
                self.size = data.size()[0]
            elif not train and not _train:
                self.test_data = data
                self.test_labels = labels
                self.test_locs = locs
                self.size = data.size()[0]

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.train:
            return self.train_data[i], self.train_labels[i], self.train_locs[i]
        else:
            return self.test_data[i], self.test_labels[i], self.test_locs[i]
