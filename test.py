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

model = models.SequentialGlimpsedClassifier(n_max=20, pre_lstm_filters=[16, 32, 64], lstm_dims=512, mlp_dims=512, n_class_embed_dims=50)
opt = T.optim.Adam(model.parameters())
#loss_fn = losses.RLClassifierLoss()
loss_fn = losses.SupervisedClassifierLoss()
ones = T.ones(batch_size, 10).long()

print(dict(model.named_parameters()).keys())
print(sum(NP.prod(p.size()) for p in model.parameters()))

for epoch in range(50):
    for i, (x, _y, B) in enumerate(mnist_dataloader):
        batch_size, n_rows, n_cols = x.size()
        y = T.LongTensor(batch_size, 10).zero_().scatter_add_(1, _y, ones)
        x = tovar(x.float() / 255)
        y = tovar(y)
        __y = tovar(_y)
        B = tovar(B)
        y_hat, y_hat_logprob, p, p_logprob = model(x.unsqueeze(1).expand(batch_size, 3, n_rows, n_cols))
        loss = loss_fn(__y[:, 0], model.y_pre, model.p_pre)
        opt.zero_grad()
        loss.backward()
        #grad_clip(model.parameters(), 1)
        opt.step()
        if i % 100 == 0:
            #print(epoch, i, tonumpy(loss_fn.r).sum(1).mean(), tonumpy(loss))
            print(epoch, i, tonumpy((__y[:, 0] == model.y_pre.max(-1)[1][:, -1]).sum()), tonumpy(loss))
