import torch as T
import torch.nn as NN
import torch.nn.functional as F
import models
import losses
from util import *
import numpy.random as RNG
from datasets import MNISTMulti

mnist = MNISTMulti('.', n_digits=1, backrand=128, image_rows=70, image_cols=70)
batch_size = 32
mnist_dataloader = T.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True, drop_last=True)

model = models.SequentialGlimpsedClassifier()
opt = T.optim.Adam(model.parameters())
loss_fn = losses.RLClassifierLoss()
ones = T.ones(batch_size, 10).long()

print(dict(model.named_parameters()).keys())

for epoch in range(50):
    for i, (x, _y, B) in enumerate(mnist_dataloader):
        batch_size, n_rows, n_cols = x.size()
        y = T.LongTensor(batch_size, 10).zero_().scatter_add_(1, _y, ones)
        x = tovar(x.float() / 255)
        y = tovar(y)
        B = tovar(B)
        for j in range(10000):
            y_hat, y_hat_logprob, p, p_logprob = model(x.unsqueeze(1).expand(batch_size, 3, n_rows, n_cols))
            loss = loss_fn(y, y_hat, y_hat_logprob, p, p_logprob)
            #loss = F.logsigmoid(model.p_pre).sum()
            opt.zero_grad()
            loss.backward()
            #print(tonumpy(loss))
            print([(name, tonumpy(T.norm(p.grad.data)), tonumpy(T.max(p.data))) for name, p in model.named_parameters() if p.grad is not None])
            grad_clip(model.parameters(), 1)
            opt.step()
            if j % 100 == 0:
                print(epoch, i, tonumpy(loss_fn.r).sum(1).mean(), tonumpy(loss))
    print('_y', tonumpy(_y))
    print('y_hat', tonumpy(y_hat)[:, :, 0])
    print('p', tonumpy(p)[:, :, 0])
