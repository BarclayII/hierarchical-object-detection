
import torch as T
import torch.nn as NN
import torch.nn.functional as F
import copy
from util import *
from models import build_mlp

class RLClassifierLoss(NN.Module):
    def __init__(self, correct=1, incorrect=-1, gamma=1, ewma=0.7):
        NN.Module.__init__(self)
        self.correct = correct
        self.incorrect = incorrect
        self.gamma = gamma
        self.ewma = ewma

    def forward(self, y, y_hat, log_y_hat, p, log_p, v_B, log_v_B):
        '''
        y: (batch_size, n_classes) LongTensor, counts
        '''
        y = y.clone()
        pb = p.byte()
        r_list = []
        log_prob_list = []
        n_steps = y_hat.size()[1]
        ones = T.ones_like(y)
        for t in range(n_steps):
            right = ((y.gather(1, y_hat[:, t]) > 0) & pb[:, t]).float()
            wrong = ((y.gather(1, y_hat[:, t]) == 0) & pb[:, t]).float()
            none = (y_hat.sum(1) == 0).float()
            y_pred = y.clone().zero_()
            y_pred.scatter_add_(1, y_hat[:, t], T.ones_like(y_pred))
            y = (y - y_pred).clamp(min=0)
            r_list.append(right * self.correct + (wrong + none) * self.incorrect)

        r = T.stack(r_list, 1)

        self.r = r
        self.b = r.mean(0, keepdim=True)

        gamma = self.gamma ** tovar(
                T.arange(n_steps)[None, :, None].expand_as(r))
        self.q = reverse(reverse(gamma * (r - self.b), 1).cumsum(1), 1)
        self.logprob = log_p + pb.float() * log_y_hat + log_v_B.mean(-1)

        loss = -(self.logprob * self.q).mean()
        return loss

class HybridClassifierLoss(NN.Module):
    def __init__(self, state_size=128, correct=1, gamma=1, ewma=0.7):
        NN.Module.__init__(self)
        self.correct = correct
        self.critic = build_mlp(input_size=state_size,
                                layer_sizes=[state_size, 1])
        self.opt = T.optim.Adam(self.critic.parameters())

    def forward(self, model, y):
        batch_size, n_steps, state_size = model.h.size()
        y_loss = F.cross_entropy(model.y_pre[:, -1], y[:, 0])
        h = model.h.detach()

        b = self.critic(h.view(-1, state_size)).view(batch_size, n_steps)
        r_list = []

        for t in range(n_steps):
            if t != n_steps - 1:
                r = tovar(T.zeros(batch_size))
            else:
                r = (model.y_hat[:, t, 0] == y[:, 0]).float() * self.correct
            r_list.append(r)

        r = T.stack(r_list, 1)

        # Only count the last step
        b_loss = ((r[:, -1] - b[:, -1]) ** 2).mean()

        v_B_loss = -(r[:, -1] - b[:, -1]) * model.v_B_logprob.mean(-1).mean(-1)
        v_B_loss = v_B_loss.mean()

        if self.training:
            self.opt.zero_grad()
            b_loss.backward(retain_graph=True)
            self.opt.step()

        return y_loss + v_B_loss

class SupervisedClassifierLoss(NN.Module):
    def forward(self, y, y_pre, p_pre):
        y_loss = F.cross_entropy(y_pre[:, -1], y)
        #p = p_pre.clone().zero_()
        #p[:, -1] = 1
        #p_loss = F.binary_cross_entropy_with_logits(p_pre, p)
        p_loss = 0
        return y_loss + p_loss
