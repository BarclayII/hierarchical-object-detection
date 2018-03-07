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
import solver

batch_size = 64
ones = T.ones(batch_size, 10).long()

def process_datum(x, y, B, volatile=False):
    y_cnt = cuda(T.LongTensor(batch_size, 10).zero_().scatter_add_(1, y, ones))
    x = tovar(x.float() / 255, volatile=volatile)
    y_cnt = tovar(y_cnt, volatile=volatile)
    y = tovar(y, volatile=volatile)
    B = tovar(B, volatile=volatile)
    _, n_rows, n_cols = x.size()
    x = x.unsqueeze(1).expand(batch_size, 3, n_rows, n_cols)

    return x, y_cnt, y, B

process_datum_valid = partial(process_datum, volatile=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n-max', type=int, default=1)
parser.add_argument('--glim-size', type=int, default=70)
parser.add_argument('--teacher', action='store_true')
args = parser.parse_args()

mnist_train = MNISTMulti('.', n_digits=1, backrand=128, image_rows=70, image_cols=70, download=True)
mnist_valid = MNISTMulti('.', n_digits=1, backrand=128, image_rows=70, image_cols=70, download=False, mode='valid')
mnist_train_dataloader = wrap_output(
        T.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0),
        process_datum)
mnist_valid_dataloader = wrap_output(
        T.utils.data.DataLoader(mnist_valid, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0),
        process_datum_valid)

if args.teacher:
    model = cuda(models.CNNClassifier(mlp_dims=512))
    def train_loss(solver):
        x, y_cnt, y, B = solver.datum
        return F.cross_entropy(solver.model.y_pre, y[:, 0])

    def acc(solver):
        x, y_cnt, y, B = solver.datum
        return NP.asscalar(tonumpy((y[:, 0] == solver.model.y_pre.max(-1)[1]).sum()))

else:
    model = cuda(models.SequentialGlimpsedClassifier(
        n_max=args.n_max,
        pre_lstm_filters=[16, 32, 64],
        lstm_dims=512,
        mlp_dims=512,
        n_class_embed_dims=50,
        glimpse_size=(args.glim_size, args.glim_size)))
    #loss_fn = losses.RLClassifierLoss()
    loss_fn = losses.SupervisedClassifierLoss()
    def train_loss(solver):
        x, y_cnt, y, B = solver.datum
        loss = loss_fn(y[:, 0], solver.model.y_pre, solver.model.p_pre)
        return loss

    def acc(solver):
        x, y_cnt, y, B = solver.datum
        return NP.asscalar(tonumpy((y[:, 0] == solver.model.y_pre.max(-1)[1][:, -1]).sum()))


def model_output(solver):
    x, y_cnt, y, B = solver.datum
    return solver.model(x)

def on_before_run(solver):
    solver.best_correct = 0

def on_before_step(solver):
    solver.norm = clip_grad_norm(solver.model_params, 1)

def on_after_train_batch(solver):
    print(solver.epoch,
          solver.batch,
          solver.eval_metric[0],
          tonumpy(solver.train_loss),
          tonumpy(solver.norm),
          max(p.data.max() for p in solver.model_params),
          min(p.data.min() for p in solver.model_params))
    if not args.teacher:
        print(tonumpy(solver.model.v_B[0]))

def on_before_eval(solver):
    solver.total = solver.correct = 0

def on_after_eval_batch(solver):
    solver.total += batch_size
    solver.correct += solver.eval_metric[0]

def on_after_eval(solver):
    print(solver.epoch, solver.correct, '/', solver.total)
    if solver.correct > solver.best_correct:
        solver.best_correct = correct
        T.save(solver.model, solver.save_path)

def run():
    sgd_gamma = 1
    sgd_lambda = 0.1

    params = [p for p in model.parameters() if p.requires_grad]
    opt = T.optim.RMSprop(params, lr=1e-5)

    print(dict(model.named_parameters()).keys())
    print(sum(NP.prod(p.size()) for p in params))

    s = solver.Solver(mnist_train_dataloader,
                      mnist_valid_dataloader,
                      model,
                      model_output,
                      train_loss,
                      [acc],
                      opt)
    s.save_path = 'teacher.pt' if args.teacher else 'model.pt'
    s.register_callback('before_run', on_before_run)
    s.register_callback('before_step', on_before_step)
    s.register_callback('after_train_batch', on_after_train_batch)
    s.register_callback('before_eval', on_before_eval)
    s.register_callback('after_eval_batch', on_after_eval_batch)
    s.register_callback('after_eval', on_after_eval)
    s.run(500)

if __name__ == '__main__':
    run()
