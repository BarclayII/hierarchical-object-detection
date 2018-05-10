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

parser = argparse.ArgumentParser()
parser.add_argument('--n-children', type=int, default=2)
parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--glim-size', type=int, default=70)
parser.add_argument('--image-size', type=int, default=70)
parser.add_argument('--teacher', action='store_true')
parser.add_argument('--env', type=str, default='main')
parser.add_argument('--backnoise', type=int, default=0)
parser.add_argument('--glim-type', type=str, default='gaussian')
parser.add_argument('--loss', type=str, default='supervised')
args = parser.parse_args()

n_classes = 11
n_digits = 3

wm = VisdomWindowManager(env=args.env)

if args.teacher:
    model = cuda(models.CNNClassifier(mlp_dims=512))
    def train_loss(solver):
        x, y_cnt, y, B = solver.datum
        B = B.float() / args.image_size
        return F.cross_entropy(solver.model.y_pre, y[:, 0])

    def acc(solver):
        global n_classes
        x, y_cnt, y, B = solver.datum
        B = B.float() / args.image_size
        y_pre = solver.model.y_pre.max(-1)[1]
        y_pre_cnt = tovar(cuda(T.LongTensor(batch_size, n_classes).zero_())).scatter_add_(1, y_pre, T.ones_like(y_pre))
        return NP.asscalar(tonumpy((y_cnt == y_pre_cnt).prod(1).sum()))

else:
    model = cuda(models.TreeGlimpsedClassifier(
        n_classes=n_classes,
        n_children=args.n_children,
        n_depth=args.depth,
        ))
    #loss_fn = losses.RLClassifierLoss()
    loss_fn = losses.SupervisedClassifierLoss()

    register_backward_hooks(model)
    register_backward_hooks(loss_fn)
    model.register_backward_hook(partial(log_grad, name='model'))
    loss_fn.register_backward_hook(partial(log_grad, name='loss_fn'))

    def train_loss(solver):
        x, y_cnt, y, B = solver.datum
        batch_size, n_objects = y.size()

        B = B.float() / args.image_size
        batch_size, n_labels = y.size()
        #loss = loss_fn(solver.model)
        loss = loss_fn(y, solver.model.y_pre)

        return loss

    def acc(solver):
        global n_classes
        x, y_cnt, y, B = solver.datum
        B = B.float() / args.image_size
        y_pre = solver.model.y_pre
        y_pre = y_pre.max(-1)[1]
        y_pre_cnt = tovar(cuda(T.LongTensor(batch_size, n_classes).zero_())).scatter_add_(1, y_pre, T.ones_like(y_pre))
        return NP.asscalar(tonumpy((y_cnt == y_pre_cnt).prod(1).sum()))


def process_datum(x, y, B, volatile=False):
    global n_classes
    batch_size, n_rows, n_cols = x.size()
    n_objects = y.size()[1]
    y = cuda(y)
    y = T.cat([
        y,
        cuda(T.zeros(batch_size, model.n_leaves - n_objects).long() + n_classes - 1),
        ], 1)
    y_cnt = cuda(T.LongTensor(batch_size, n_classes).zero_()).scatter_add_(1, y, T.ones_like(y))
    x = tovar(x.float() / 255, volatile=volatile)
    y_cnt = tovar(y_cnt, volatile=volatile)
    y = tovar(y, volatile=volatile)
    B = tovar(B, volatile=volatile)
    x = x.unsqueeze(1).expand(batch_size, 3, n_rows, n_cols)

    return x, y_cnt, y, B

process_datum_valid = partial(process_datum, volatile=True)

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


def model_output(solver):
    x, y_cnt, y, B = solver.datum
    B = B.float() / args.image_size
    return solver.model(x, y=y_cnt)

def on_before_run(solver):
    solver.best_correct = 0

def on_before_train(solver):
    if not args.teacher:
        loss_fn.train()

def on_before_step(solver):
    solver.norm = clip_grad_norm(solver.model_params, 1)

def on_after_train_batch(solver):
    x, y_cnt, y, B = solver.datum
    B = B.float() / args.image_size
    print(solver.epoch,
          solver.batch,
          solver.eval_metric[0],
          tonumpy(solver.train_loss),
          tonumpy(solver.norm),
          max(p.data.max() for p in solver.model_params),
          min(p.data.min() for p in solver.model_params))
    if not args.teacher:
        print(tonumpy(solver.model.v_B[0]))
        print(tonumpy(B[0]))
        print('IOU', tonumpy(iou(
            solver.model.v_B[:, :, :4].index_select(
                1,
                tovar(T.LongTensor([3, 4, 5]))),
            B)).mean())
        #print('R', tonumpy(loss_fn.r)[0])
        #print('B', tonumpy(loss_fn.b)[0])
        #print('Q', tonumpy(loss_fn.q)[0])

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
            fig, ax = PL.subplots(2, 5)
            fig.set_size_inches(10, 8)

            x, _, _, B = solver.datum
            ax.flatten()[0].imshow(tonumpy(x[0].permute(1, 2, 0)))
            addbox(ax.flatten()[0], tonumpy(B[0, 0]), 'red')
            v_B = tonumpy(solver.model.v_B)
            for i in range(solver.model.n_nodes):
                addbox(ax.flatten()[0], tonumpy(v_B[0, i, :4] * args.image_size), 'yellow', i+1)
                #ax.flatten()[i + 1].imshow(tonumpy(solver.model.g[0, i].permute(1, 2, 0)), vmin=0, vmax=1)
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
