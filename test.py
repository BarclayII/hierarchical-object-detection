import torch as T
import torch.nn as NN
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import models
import losses
from util import *
import numpy.random as RNG
from datasets import MNISTMulti, wrap_output
import numpy as NP
from timer import Timer
import argparse
from functools import partial

def process_datum(x, _y, B, volatile=False):
    y = cuda(T.LongTensor(batch_size, 10).zero_().scatter_add_(1, _y, ones_))
    x = tovar(x.float() / 255, volatile=volatile)
    y = tovar(y, volatile=volatile)
    __y = tovar(_y, volatile=volatile)
    B = tovar(B, volatile=volatile)

    return x, y, __y, B
process_datum_valid = partial(process_datum, volatile=True)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-max', type=int, default=1)
    parser.add_argument('--glim-size', type=int, default=70)
    args = parser.parse_args()

    mnist_train = MNISTMulti('.', n_digits=1, backrand=128, image_rows=70, image_cols=70, download=True)
    mnist_valid = MNISTMulti('.', n_digits=1, backrand=128, image_rows=70, image_cols=70, download=False, mode='valid')
    batch_size = 128
    mnist_train_dataloader = wrap_output(
            T.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0),
            process_datum)
    mnist_valid_dataloader = wrap_output(
            T.utils.data.DataLoader(mnist_valid, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0),
            process_datum_valid)

    model = cuda(models.SequentialGlimpsedClassifier(
        n_max=args.n_max,
        pre_lstm_filters=[16, 32, 64],
        lstm_dims=512,
        mlp_dims=512,
        n_class_embed_dims=50,
        glimpse_size=(args.glim_size, args.glim_size)))
    #loss_fn = losses.RLClassifierLoss()
    loss_fn = losses.SupervisedClassifierLoss()
    ones = T.ones(batch_size, 10).long()
    sgd_gamma = 1
    sgd_lambda = 0.1

    params = [p for p in model.parameters() if p.requires_grad]

    print(dict(model.named_parameters()).keys())
    print(sum(NP.prod(p.size()) for p in params))

    for epoch in range(100):
        #opt = T.optim.SGD(params, lr=1 / (1 + 1e-1 * epoch))
        opt = T.optim.RMSprop(params, lr=1e-5)
        for i, (x, y, __y, B) in enumerate(mnist_train_dataloader):
            batch_size, n_rows, n_cols = x.size()
            with Timer.new('forward', print_=True):
                y_hat, y_hat_logprob, p, p_logprob = model(x.unsqueeze(1).expand(batch_size, 3, n_rows, n_cols))
                loss = loss_fn(__y[:, 0], model.y_pre, model.p_pre)
                reg = sum(p.norm() ** 2 for p in params) * 1e-4
            opt.zero_grad()
            with Timer.new('backward', print_=True):
                #(loss + reg).backward()
                loss.backward()
            #norm = 0
            #for p in params:
            #    if p.grad is not None:
            #        norm += p.grad.norm()
            #        p.grad.data.clamp_(min=-1, max=1)
            with Timer.new('clip', print_=True):
                norm = clip_grad_norm(params, 1)
            opt.step()
            if i % 1 == 0:
                #print(epoch, i, tonumpy(loss_fn.r).sum(1).mean(), tonumpy(loss))
                print(epoch, i, tonumpy((__y[:, 0] == model.y_pre.max(-1)[1][:, -1]).sum()), tonumpy(loss),
                      tonumpy(reg), tonumpy(norm),
                      max(p.data.max() for p in params),
                      min(p.data.min() for p in params))
                print(tonumpy(model.v_B[0]))

        total = correct = 0
        for i, (x, y, __y, B) in enumerate(mnist_valid_dataloader):
            batch_size, n_rows, n_cols = x.size()
            y_hat, y_hat_logprob, p, p_logprob = model(x.unsqueeze(1).expand(batch_size, 3, n_rows, n_cols))
            total += batch_size
            correct += NP.asscalar(tonumpy((__y[:, 0] == model.y_pre.max(-1)[1][:, -1]).sum()))
        print(epoch, correct, '/', total)

if __name__ == '__main__':
    run()
