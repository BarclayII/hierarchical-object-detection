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
import argparse
from functools import partial
import solver
from viz import VisdomWindowManager
import matplotlib.pyplot as PL
from logger import register_backward_hooks, log_grad

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
parser.add_argument('--image-size', type=int, default=70)
parser.add_argument('--teacher', action='store_true')
parser.add_argument('--env', type=str, default='main')
parser.add_argument('--backnoise', type=int, default=0)
parser.add_argument('--glim-type', type=str, default='gaussian')
parser.add_argument('--loss', type=str, default='supervised')
args = parser.parse_args()

n_digits = 1 if args.loss != 'multi' else 3

mnist_train = MNISTMulti('.', n_digits=n_digits, backrand=args.backnoise,
        image_rows=args.image_size, image_cols=args.image_size, download=True)
mnist_valid = MNISTMulti('.', n_digits=n_digits, backrand=args.backnoise,
        image_rows=args.image_size, image_cols=args.image_size, download=False, mode='valid')
mnist_train_dataloader = wrap_output(
        T.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0),
        process_datum)
mnist_valid_dataloader = wrap_output(
        T.utils.data.DataLoader(mnist_valid, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0),
        process_datum_valid)

wm = VisdomWindowManager(env=args.env)

if args.teacher:
    model = cuda(models.CNNClassifier(mlp_dims=512))
    def train_loss(solver):
        x, y_cnt, y, B = solver.datum
        B = B.float() / args.image_size
        return F.cross_entropy(solver.model.y_pre, y[:, 0])

    def acc(solver):
        x, y_cnt, y, B = solver.datum
        B = B.float() / args.image_size
        y_pre = solver.model.y_pre.max(-1)[1]
        y_pre_cnt = tovar(cuda(T.LongTensor(batch_size, 10).zero_())).scatter_add_(1, y_pre, T.ones_like(y_pre))
        return NP.asscalar(tonumpy((y_cnt == y_pre_cnt).prod(1).sum()))

else:
    model = cuda(models.SequentialGlimpsedClassifier(
        n_max=args.n_max,
        pre_lstm_filters=[5, 5, 10],
        lstm_dims=512,
        mlp_dims=512,
        n_class_embed_dims=50,
        relative_previous=False,
        glimpse_size=(args.glim_size, args.glim_size),
        glimpse_type=args.glim_type,
        glimpse_sample=(args.loss == 'hybrid')))

    if args.loss == 'supervised':
        loss_fn = losses.SupervisedClassifierLoss()
    elif args.loss == 'hybrid':
        teacher = T.load('teacher.pt')
        loss_fn = losses.HybridClassifierLoss(state_size=512, teacher=teacher)  # same as LSTM
    elif args.loss == 'multi':
        loss_fn = losses.SupervisedMultitaskMultiobjectLoss()

    register_backward_hooks(model)
    register_backward_hooks(loss_fn)
    model.register_backward_hook(partial(log_grad, name='model'))
    loss_fn.register_backward_hook(partial(log_grad, name='loss_fn'))

    def train_loss(solver):
        x, y_cnt, y, B = solver.datum
        B = B.float() / args.image_size
        batch_size, n_labels = y.size()
        if args.loss == 'supervised':
            loss = loss_fn(y[:, 0], solver.model.y_pre, solver.model.p_pre)
        elif args.loss == 'hybrid':
            loss = loss_fn(solver.model, y)
        elif args.loss == 'map':
            y = T.cat(
                [y, tovar(T.zeros(batch_size, args.n_max - n_labels).long() + mnist_train.n_classes)],
                -1
                )
            loss = loss_fn(y, solver.model.y_pre, solver.model.p_pre)
        elif args.loss == 'multi':
            loss = loss_fn(y, solver.model.y_pre, B, solver.model.v_B_pre, solver.model.idx)
        return loss

    def acc(solver):
        x, y_cnt, y, B = solver.datum
        B = B.float() / args.image_size
        if args.loss != 'multi':
            y_pre = solver.model.y_pre.max(-1)[1]
        else:
            y_pre = solver.model.y_pre.max(-1)[1][:, 1:]
        y_pre_cnt = tovar(cuda(T.LongTensor(batch_size, 10).zero_())).scatter_add_(1, y_pre, T.ones_like(y_pre))
        return NP.asscalar(tonumpy((y_cnt == y_pre_cnt).prod(1).sum()))


def model_output(solver):
    x, y_cnt, y, B = solver.datum
    B = B.float() / args.image_size
    if args.loss == 'multi':
        return solver.model(x, y=y, B=B, feedback='oracle')
    return solver.model(x, y=y)

def on_before_run(solver):
    solver.best_correct = 0

def on_before_train(solver):
    if not args.teacher:
        loss_fn.train()

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
        if args.loss == 'hybrid':
            print('R', tonumpy(loss_fn.r)[0])
            print('ΔR', tonumpy(loss_fn.dr)[0])
            print('B', tonumpy(loss_fn.b)[0])
            print('Q', tonumpy(loss_fn.q)[0])

def on_before_eval(solver):
    solver.total = solver.correct = 0

    if not args.teacher:
        solver.nviz = 10
        loss_fn.eval()

def on_after_eval_batch(solver):
    solver.total += batch_size
    solver.correct += solver.eval_metric[0]

    if not args.teacher:
        if solver.nviz > 0:
            solver.nviz -= 1
            if args.loss == 'hybrid':
                fig, ax = PL.subplots(2, 5)
            else:
                fig, ax = PL.subplots(2, 3)
            fig.set_size_inches(10, 8)

            x, _, _, B = solver.datum
            ax.flatten()[0].imshow(tonumpy(x[0].permute(1, 2, 0)))
            addbox(ax.flatten()[0], tonumpy(B[0, 0]), 'red')
            v_B = tonumpy(solver.model.v_B)
            for i in range(args.n_max):
                addbox(ax.flatten()[0], tonumpy(v_B[0, i, :4] * args.image_size), 'yellow', i+1)
                ax.flatten()[i + 1].imshow(tonumpy(solver.model.g[0, i].permute(1, 2, 0)), vmin=0, vmax=1)
                if args.loss == 'hybrid' and i < args.n_max - 1:
                    ax.flatten()[i + 6].imshow(tonumpy(loss_fn.m[0, i].permute(1, 2, 0).clamp(min=0, max=1)),
                            vmin=0, vmax=1)
                    ax.flatten()[i + 6].set_title('%.3f' % NP.asscalar(tonumpy(loss_fn.r[0, i])))
            wm.display_mpl_figure(fig, win='viz{}'.format(solver.nviz))

def on_after_eval(solver):
    print(solver.epoch, solver.correct, '/', solver.total)
    if solver.correct > solver.best_correct:
        solver.best_correct = solver.correct
        T.save(solver.model, solver.save_path)

def run():
    params = [p for p in model.parameters() if p.requires_grad]
    opt = T.optim.RMSprop(params, lr=1e-4)

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
    s.register_callback('before_train', on_before_train)
    s.register_callback('before_step', on_before_step)
    s.register_callback('after_train_batch', on_after_train_batch)
    s.register_callback('before_eval', on_before_eval)
    s.register_callback('after_eval_batch', on_after_eval_batch)
    s.register_callback('after_eval', on_after_eval)
    s.run(500)

if __name__ == '__main__':
    run()
