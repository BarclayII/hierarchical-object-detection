import torch as T
import torch.nn as NN
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import models
import losses
from util import *
import numpy.random as RNG
from datasets import MNISTMulti
import numpy as NP

import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-max', type=int, default=1)
    parser.add_argument('--glim-size', type=int, default=70)
    args = parser.parse_args()

    mnist_train = MNISTMulti('.', n_digits=1, backrand=128, image_rows=70, image_cols=70, download=True)
    mnist_valid = MNISTMulti('.', n_digits=1, backrand=128, image_rows=70, image_cols=70, download=False, mode='valid')
    batch_size = 32
    mnist_train_dataloader = T.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    mnist_valid_dataloader = T.utils.data.DataLoader(mnist_valid, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    model = cuda(models.SequentialGlimpsedClassifier(n_max=args.n_max, pre_lstm_filters=[16, 32, 64, 128, 128], lstm_dims=512, mlp_dims=512, n_class_embed_dims=50, glimpse_size=(args.glim_size, args.glim_size)))
    #loss_fn = losses.RLClassifierLoss()
    loss_fn = losses.SupervisedClassifierLoss()
    ones = T.ones(batch_size, 10).long()
    sgd_gamma = 1
    sgd_lambda = 0.1

    print(dict(model.named_parameters()).keys())
    print(sum(NP.prod(p.size()) for p in model.parameters()))

    for epoch in range(100):
        #opt = T.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1 / (1 + 1e-1 * epoch))
        opt = T.optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        for i, (x, _y, B) in enumerate(mnist_train_dataloader):
            batch_size, n_rows, n_cols = x.size()
            y = cuda(T.LongTensor(batch_size, 10).zero_().scatter_add_(1, _y, ones))
            x = tovar(x.float() / 255)
            y = tovar(y)
            __y = tovar(_y)
            B = tovar(B)
            y_hat, y_hat_logprob, p, p_logprob = model(x.unsqueeze(1).expand(batch_size, 3, n_rows, n_cols))
            loss = loss_fn(__y[:, 0], model.y_pre, model.p_pre)
            reg = sum((p ** 2).sum() for p in model.parameters()) * 1e-4
            opt.zero_grad()
            (loss + reg).backward()
            #norm = 0
            #for p in model.parameters():
            #    if p.grad is not None:
            #        norm += p.grad.norm()
            #        p.grad.data.clamp_(min=-1, max=1)
            norm = clip_grad_norm(model.parameters(), 1)
            opt.step()
            if i % 1 == 0:
                #print(epoch, i, tonumpy(loss_fn.r).sum(1).mean(), tonumpy(loss))
                print(epoch, i, tonumpy((__y[:, 0] == model.y_pre.max(-1)[1][:, -1]).sum()), tonumpy(loss),
                      tonumpy(reg), tonumpy(norm),
                      max(p.data.max() for p in model.parameters()),
                      min(p.data.min() for p in model.parameters()))
                print(tonumpy(model.v_B[0]))

        total = correct = 0
        for i, (x, _y, B) in enumerate(mnist_valid_dataloader):
            batch_size, n_rows, n_cols = x.size()
            y = cuda(T.LongTensor(batch_size, 10).zero_().scatter_add_(1, _y, ones))
            x = tovar(x.float() / 255, volatile=True)
            y = tovar(y, volatile=True)
            __y = tovar(_y, volatile=True)
            B = tovar(B)
            y_hat, y_hat_logprob, p, p_logprob = model(x.unsqueeze(1).expand(batch_size, 3, n_rows, n_cols))
            total += batch_size
            correct += NP.asscalar(tonumpy((__y[:, 0] == model.y_pre.max(-1)[1][:, -1]).sum()))
        print(epoch, correct, '/', total)

if __name__ == '__main__':
    run()
