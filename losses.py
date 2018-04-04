
import torch as T
import torch.nn as NN
import torch.nn.functional as F
import copy
from util import *
from models import build_mlp
import numpy as NP
from scipy.optimize import linear_sum_assignment

class RLClassifierLoss(NN.Module):
    def __init__(self, gamma=1, ewma=0.7):
        NN.Module.__init__(self)
        self.gamma = gamma
        self.ewma = ewma

    def forward(self, model):
        '''
        y: (batch_size, n_classes) LongTensor, counts
        '''
        r_list = []
        log_prob_list = []
        n_steps = y_hat.size()[1]
        ones = T.ones_like(y)
        for node in model.T:
            r_list.append(node.R)
            log_prob_list.append(node.y_pre.gather(1, node.y_hat))

        r = T.stack(r_list, 1)

        self.r = r
        self.b = r.mean(0, keepdim=True)
        self.logprob = T.stack(T.stack(log_prob_list, 1))[..., 0]

        gamma = self.gamma ** tovar(
                T.arange(n_steps)[None, :, None].expand_as(r))
        self.q = reverse(reverse(gamma * (r - self.b), 1).cumsum(1), 1)

        loss = -(self.logprob * self.q).mean()
        return loss

class HybridClassifierLoss(NN.Module):
    def __init__(self, state_size=128, correct=1, gamma=1, ewma=0.7, input_size=(3, 70, 70), teacher=None):
        NN.Module.__init__(self)
        self.correct = correct
        self.critic = cuda(build_mlp(input_size=state_size,
                                     layer_sizes=[state_size, 1]))
        self.opt = T.optim.Adam(self.critic.parameters())
        self.input_size = input_size
        self.teacher = teacher
        self.gamma = gamma
        for p in teacher.parameters():
            p.requires_grad = False

    def forward(self, model, y, critic=True):
        batch_size, n_steps, state_size = model.h.size()
        y_loss = F.cross_entropy(model.y_pre[:, -1], y[:, 0])
        h = model.h.detach()

        if critic:
            b = self.critic(h.view(-1, state_size)).view(batch_size, n_steps)
        else:
            b = 0
        dr_list = []
        m_list = []
        m = tovar(T.zeros(batch_size, *self.input_size))
        prev_r = 0

        for t in range(n_steps):
            if self.teacher is None:
                if t != n_steps - 1:
                    r = tovar(T.zeros(batch_size))
                else:
                    r = (model.y_hat[:, t, 0] == y[:, 0]).float() * self.correct
                dr_list.append(r)
            else:
                m = overlay(model.g[:, t], model.v_B[:, t, :4], m)
                m_list.append(m)
                y_teacher_hat = self.teacher(m)
                r = y_teacher_hat.gather(1, y[:, 0:1])[:, 0].detach()
                dr_list.append(r - prev_r)
                prev_r = r
                #r = (y_teacher == y[:, 0]).float() * self.correct

        r = T.stack(dr_list, 1)
        m = T.stack(m_list, 1)

        if self.teacher is None:
            # Only count the last step
            b_loss = ((r[:, -1] - b[:, -1]) ** 2).mean()
            v_B_loss = -(r[:, -1] - b[:, -1]) * model.v_B_logprob.mean(-1).mean(-1)
        else:
            b_loss = ((r - b) ** 2).mean()
            gamma = self.gamma ** tovar(
                    T.arange(n_steps)[None, :].expand_as(r))
            self.q = reverse(reverse(gamma * (r - b), 1).cumsum(1), 1)
            v_B_loss = -(self.q * model.v_B_logprob.mean(-1)).mean(-1)
        v_B_loss = v_B_loss.mean()

        if self.training and critic:
            self.opt.zero_grad()
            b_loss.backward(retain_graph=True)
            self.opt.step()
            print(max(p.data.max() for p in self.critic.parameters()))
            print(min(p.data.min() for p in self.critic.parameters()))

        self.b = b
        self.dr = r
        self.r = T.cumsum(r, 1)
        self.m = m

        #return y_loss + v_B_loss
        return v_B_loss

class SupervisedClassifierLoss(NN.Module):
    def forward(self, y, y_pre, p_pre):
        y_loss = F.cross_entropy(y_pre[:, -1], y)
        #p = p_pre.clone().zero_()
        #p[:, -1] = 1
        #p_loss = F.binary_cross_entropy_with_logits(p_pre, p)
        p_loss = 0
        return y_loss + p_loss


class SupervisedMultitaskMultiobjectLoss(NN.Module):
    def forward(self, y, y_pre, B, B_pre, idx):
        batch_size, n_steps, n_classes = y_pre.size()

        y = y.gather(1, idx[:, :, 0]).view(-1)
        y_pre = y_pre[:, 1:].contiguous().view(-1, n_classes)
        B = B.gather(1, idx.expand(batch_size, n_steps - 1, 4))
        B_pre = B_pre[:, :-1, :4]

        y_loss = F.cross_entropy(y_pre, y)
        B_loss = ((B - B_pre) ** 2).mean()

        return y_loss + B_loss
