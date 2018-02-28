
import torch as T
import numpy as NP
import os

USE_CUDA = os.getenv('USE_CUDA', None)

def cuda(x, device=None, async_=False):
    return x.cuda() if USE_CUDA else x

def tovar(x, *args, dtype='float32', **kwargs):
    if not T.is_tensor(x):
        x = T.from_numpy(NP.array(x, dtype=dtype))
    return T.autograd.Variable(cuda(x), *args, **kwargs)

def tonumpy(x):
    if isinstance(x, T.autograd.Variable):
        x = x.data
    if T.is_tensor(x):
        x = x.cpu().numpy()
    return x

def create_onehot(idx, size):
    onehot = tovar(T.zeros(*size))
    onehot = onehot.scatter(1, idx.unsqueeze(1), 1)
    return onehot

def reverse(x, dim):
    idx = T.arange(x.size()[dim] - 1, -1, -1).long()
    if isinstance(x, T.autograd.Variable):
        idx = tovar(idx)
    return x.index_select(dim, idx)

def addbox(ax, b, ec):
    import matplotlib.patches as PA
    ax.add_patch(PA.Rectangle((b[0] - b[2] / 2, b[1] - b[3] / 2), b[2], b[3],
                 ec=ec, fill=False, lw=1))

def grad_clip(params, max_l2):
    norm = sum((p.grad.data ** 2).sum() for p in params if p.grad is not None) ** 0.5
    if norm > max_l2:
        for p in params:
            if p.grad is not None:
                p.grad.data /= norm / max_l2
    return norm
