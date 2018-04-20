
import torch as T
import torch.nn as NN
import torch.nn.init as INIT
import torch.nn.functional as F
import numpy as NP
import numpy.random as RNG
from util import *
from glimpse import create_glimpse
from zoneout import ZoneoutLSTMCell
from collections import namedtuple

def build_cnn(**config):
    cnn_list = []
    filters = config['filters']
    kernel_size = config['kernel_size']
    in_channels = config.get('in_channels', 3)
    final_pool_size = config['final_pool_size']

    for i in range(len(filters)):
        module = NN.Conv2d(
            in_channels if i == 0 else filters[i-1],
            filters[i],
            kernel_size,
            padding=tuple((_ - 1) // 2 for _ in kernel_size),
            )
        INIT.xavier_uniform(module.weight)
        INIT.constant(module.bias, 0)
        cnn_list.append(module)
        if i < len(filters) - 1:
            cnn_list.append(NN.LeakyReLU())
    cnn_list.append(NN.AdaptiveMaxPool2d(final_pool_size))

    return NN.Sequential(*cnn_list)

def build_mlp(**config):
    mlp_list = []
    input_size = config['input_size']
    layer_sizes = config['layer_sizes']

    for i in range(len(layer_sizes)):
        module = NN.Linear(
            input_size if i == 0 else layer_sizes[i-1],
            layer_sizes[i],
            )
        INIT.xavier_uniform(module.weight)
        INIT.constant(module.bias, 0)
        mlp_list.append(module)
        if i < len(layer_sizes) - 1:
            mlp_list.append(NN.LeakyReLU())

    return NN.Sequential(*mlp_list)

class CNNClassifier(NN.Module):
    def __init__(self,
                 filters=[16, 32, 64, 128, 256],
                 kernel_size=(3, 3),
                 final_pool_size=(2, 2),
                 in_channels=3,
                 mlp_dims=128,
                 n_classes=10,
                 ):
        NN.Module.__init__(self)
        self.cnn = build_cnn(
                filters=filters,
                kernel_size=kernel_size,
                final_pool_size=final_pool_size,
                )
        self.mlp = build_mlp(
                input_size=filters[-1] * NP.asscalar(NP.prod(final_pool_size)),
                layer_sizes=[mlp_dims, n_classes])

    def forward(self, x):
        batch_size = x.size()[0]
        self.y_pre = self.mlp(self.cnn(x).view(batch_size, -1))
        self.y_hat_logprob = F.log_softmax(self.y_pre, -1)
        return self.y_hat_logprob.exp()

class SequentialGlimpsedClassifier(NN.Module):
    '''
    The RNN takes the feature maps from image glimpses as inputs for
    each time step.
    '''
    def __init__(self,
                 pre_lstm_filters=[5, 5, 10],
                 lstm_dims=128,
                 kernel_size=(3, 3),
                 final_pool_size=(2, 2),
                 n_max=10,
                 in_channels=3,
                 mlp_dims=128,
                 n_classes=10,
                 n_class_embed_dims=10,
                 glimpse_type='gaussian',
                 glimpse_size=(10, 10),
                 relative_previous=False,
                 glimpse_sample=False,
                 ):
        NN.Module.__init__(self)
        self.glimpse_type = glimpse_type
        self.glimpse = create_glimpse(glimpse_type, glimpse_size)
        self.cnn = build_cnn(
                filters=pre_lstm_filters,
                kernel_size=kernel_size,
                final_pool_size=final_pool_size,
                )
        #for p in self.cnn.parameters():
        #    p.requires_grad = False
        self.lstm = ZoneoutLSTMCell(
                pre_lstm_filters[-1] * NP.asscalar(NP.prod(final_pool_size)) +
                n_class_embed_dims + self.glimpse.att_params,
                lstm_dims,
                )
        #INIT.xavier_uniform(self.lstm.weight_ih)
        #INIT.orthogonal(self.lstm.weight_hh)
        #INIT.constant(self.lstm.bias_ih, 0)
        #INIT.constant(self.lstm.bias_hh, 0)
        #INIT.constant(self.lstm.bias_ih[lstm_dims:2*lstm_dims], 1)
        #INIT.constant(self.lstm.bias_hh[lstm_dims:2*lstm_dims], 1)
        self.proj_y = build_mlp(input_size=lstm_dims,
                                layer_sizes=[mlp_dims, n_classes])
        self.proj_p = build_mlp(input_size=lstm_dims,
                                layer_sizes=[mlp_dims, 1])
        self.proj_B = build_mlp(input_size=lstm_dims,
                                layer_sizes=[mlp_dims, self.glimpse.att_params])
        self.y_in = NN.Embedding(n_classes, n_class_embed_dims)

        self.n_max = n_max
        self.lstm_dims = lstm_dims
        self.n_classes = n_classes
        self.n_class_embed_dims = n_class_embed_dims
        self.glimpse_sample = glimpse_sample

        self.relative_previous = relative_previous

    def forward(self, x, y=None, B=None, feedback='sample'):
        '''
        x: (batch_size, nchannels, nrows, ncols)
        '''
        batch_size = x.size()[0]

        v_B = (self.glimpse.full().unsqueeze(0)
               .expand(batch_size, self.glimpse.att_params))
        v_B_logprob = T.zeros_like(v_B)
        y_emb = tovar(T.zeros(batch_size, self.n_class_embed_dims))
        s = self.lstm.zero_state(batch_size)
        if y is not None:
            y = y.clone()

        y_pre_list = []
        h_list = []
        p_pre_list = []
        v_B_list = []
        v_B_pre_list = []
        v_B_logprob_list = []
        g_list = []
        y_hat_list = []
        y_hat_logprob_list = []
        p_list = []
        p_logprob_list = []
        idx_list = []

        for t in range(self.n_max):
            v_B_list.append(v_B)
            v_B_logprob_list.append(v_B_logprob)
            g = self.glimpse(x, v_B[:, None])[:, 0]
            v_s = self.cnn(g).view(batch_size, -1)
            in_ = T.cat([v_s, y_emb, v_B], 1)
            h, s = self.lstm(in_, s)
            h_list.append(h)
            y_pre = self.proj_y(h)
            p_pre = self.proj_p(h)
            v_B_pre = self.proj_B(h)
            v_B_pre, v_B_logprob = self.glimpse.rescale(
                    v_B_pre, self.glimpse_sample)

            if self.relative_previous:
                v_B = self.glimpse.relative_to_absolute(v_B_pre, v_B)
            else:
                v_B = v_B_pre

            y_pre_list.append(y_pre)
            p_pre_list.append(p_pre)
            v_B_pre_list.append(v_B_pre)
            g_list.append(g)

            p_logprob = F.logsigmoid(p_pre)
            p = p_logprob.exp().bernoulli()
            p_list.append(p)
            p_logprob = F.logsigmoid((p * 2 - 1) * p_pre)
            p_logprob_list.append(p_logprob)
            if feedback == 'sample':
                y_hat_logprob = F.log_softmax(y_pre, -1)
                y_hat = y_hat_logprob.exp().multinomial()
                y_hat_list.append(y_hat)
                y_hat_logprob = y_hat_logprob.gather(1, y_hat)
                y_hat_logprob_list.append(y_hat_logprob)
                #y_emb = self.y_in(y_hat[:, 0]) * 0
            elif feedback == 'oracle':
                assert self.glimpse_type == 'gaussian'

                if t < self.n_max - 1:
                    n_y = (y != -1) # -1 means that we have chosen this
                    idx = n_y.float().multinomial(1)
                    _idx = idx.unsqueeze(2).expand(batch_size, 1, 4)

                if t == 0:
                    y_hat = tovar(T.zeros(batch_size, 1).long())
                    if self.training and NP.random.randint(3) > 0:
                        B_next = B.gather(1, _idx)[:, 0]
                    else:
                        B_next = v_B[:, :4]
                    idx_list.append(idx)
                    prev_y = y.gather(1, idx)
                    v_B = T.cat([B_next, v_B[:, -2:]], 1)
                    y.scatter_(1, idx, -1)
                elif t == self.n_max - 1:
                    y_hat = prev_y
                else:
                    y_hat = prev_y
                    if self.training:
                        B_next = B.gather(1, _idx)[:, 0]
                    else:
                        B_next = v_B[:, :4]
                    idx_list.append(idx)
                    prev_y = y.gather(1, idx)
                    y_emb = self.y_in(y_hat[:, 0])
                    v_B = T.cat([B_next, v_B[:, -2:]], 1)
                    y.scatter_(1, idx, -1)

                y_hat_list.append(y_hat)

        self.h = T.stack(h_list, 1)
        self.y_pre = T.stack(y_pre_list, 1)
        self.p_pre = T.stack(p_pre_list, 1)
        self.v_B = T.stack(v_B_list, 1)
        self.v_B_pre = T.stack(v_B_pre_list, 1)
        self.g = T.stack(g_list, 1)
        self.y_hat = T.stack(y_hat_list, 1)
        self.p = T.stack(p_list, 1)
        self.p_logprob = T.stack(p_logprob_list, 1)

        if self.glimpse_sample:
            self.v_B_logprob = T.stack(v_B_logprob_list, 1)

        if feedback == 'sample':
            self.y_hat_logprob = T.stack(y_hat_logprob_list, 1)
        elif feedback == 'oracle':
            self.idx = T.stack(idx_list, 1)
            self.y_hat_logprob = 0

        return self.y_hat, self.y_hat_logprob, self.p, self.p_logprob


class _Node(object):
    pass

class FixedFullTreeGlimpsedClassifier(NN.Module):
    '''
    Always generates a full tree with the given depth and number of children
    '''
    def __init__(self,
                 n_children=2,
                 depth=2,
                 pre_lstm_filters=[5, 5, 10],
                 message_dims=128,
                 lstm_dims=128,
                 kernel_size=(3, 3),
                 final_pool_size=(2, 2),
                 in_channels=3,
                 mlp_dims=128,
                 n_classes=10,
                 glimpse_type='gaussian',
                 glimpse_size=(10, 10),
                 relative_previous=False,
                 glimpse_sample=False,
                 y_sample='proportional',
                 ):
        assert glimpse_type == 'gaussian'

        NN.Module.__init__(self)

        self.glimpse_type = glimpse_type
        self.glimpse = create_glimpse(glimpse_type, glimpse_size)

        self.cnn = build_cnn(
                filters=pre_lstm_filters,
                kernel_size=kernel_size,
                final_pool_size=final_pool_size,
                )

        self.rnn_d = ZoneoutLSTMCell(
                message_dims,
                lstm_dims,
                )
        self.rnn_b = ZoneoutLSTMCell(
                message_dims,
                lstm_dims,
                )

        self.proj_I = build_mlp(
                input_size=pre_lstm_filters[-1] * NP.asscalar(NP.prod(final_pool_size)) + self.glimpse.att_params,
                layer_sizes=[mlp_dims, message_dims]
                )
        self.proj_h_y = build_mlp(input_size=lstm_dims,
                                  layer_sizes=[mlp_dims, n_classes])
        self.proj_h_B = build_mlp(input_size=lstm_dims,
                                  layer_sizes=[mlp_dims, self.glimpse.att_params])
        self.y_in = NN.Embedding(n_classes, message_dims)
        INIT.uniform(self.y_in.weight, -0.01, 0.01)

        self.lstm_dims = lstm_dims
        self.message_dims = message_dims
        self.n_classes = n_classes
        self.glimpse_sample = glimpse_sample
        self.y_sample = y_sample

        self.n_children = n_children
        self.depth = depth

        self.relative_previous = relative_previous

    @property
    def n_nodes(self):
        return (self.n_children ** self.depth - 1) // (self.n_children - 1)

    @property
    def n_leaves(self):
        return self.n_children ** (self.depth - 1)

    def isleaf(self, i):
        return i >= (self.n_children ** (self.depth - 1) - 1) // (self.n_children - 1)

    def parent(self, i):
        return (i - 1) // self.n_children

    def _dive(self, i, x, B, s, y=None):
        '''
        i: int: node index
        x: (batch_size, n_channels, n_rows, n_cols): image
        B: (batch_size, n_att_params): glimpse attention
        s: depth RNN state
        I: (batch_size, message_dims): inbound message state
        y: (batch_size, n_classes): remaining label multi set
        '''
        batch_size, n_classes = y.size()

        g = self.glimpse(x, B[:, None])[:, 0]
        phi = self.cnn(g).view(batch_size, -1)
        I = self.proj_I(T.cat([phi, B], 1))

        h, s = self.rnn_d(I, s)
        #h = s
        #h = F.relu(h + I)

        self.T[i].g = g
        self.T[i].phi = phi
        self.T[i].B = B

        if self.isleaf(i):
            y_pre = self.proj_h_y(h)
            y_logprob = F.log_softmax(y_pre)
            y_hat = y_logprob.exp().multinomial(1)
            h = self.y_in(y_hat)[:, 0]
            if y is not None:
                remaining = y.gather(1, y_hat)
                correct = (remaining > 0).float()
                wrong = (remaining == 0).float()
                R = correct * 1 + wrong * (-1)
                y = y - T.zeros_like(y).scatter_add(1, y_hat, T.ones_like(y_hat))
                y = y.clamp(min=0)
                self.T[i].R = R[:, 0]

            self.T[i].y_pre = y_pre
            self.T[i].y_logprob = y_logprob
            self.T[i].y_hat = y_hat
            return h, y
        else:
            O = []

            for _j in range(self.n_children):
                j = i * self.n_children + _j + 1

                B_pre = self.proj_h_B(h)
                B_pre, B_logprob = self.glimpse.rescale(
                        B_pre, self.glimpse_sample)
                if self.relative_previous:
                    B = self.glimpse.relative_to_absolute(B_pre, B)
                else:
                    B = B_pre

                #o, y = self._dive(j, x, B, h, y)
                o, y = self._dive(j, x, B, s, y)
                h, s = self.rnn_b(o, s)
                O.append(o)
                #h = F.relu(h + o)
            return T.stack(O, 0).mean(0), y

    def forward(self, x, y=None):
        batch_size = x.size()[0]
        B = (self.glimpse.full().unsqueeze(0)
             .expand(batch_size, self.glimpse.att_params))

        self.T = [_Node() for _ in range(0, self.n_nodes)]
        s = self.rnn_d.zero_state(batch_size)
        #h = tovar(T.zeros(batch_size, self.message_dims))

        result = self._dive(0, x, B, s, y)
        #result = self._dive(0, x, B, h, y)

        self.v_B = T.stack([node.B for node in self.T if hasattr(node, 'B')], 1)
        self.y_pre = T.stack([node.y_pre for node in self.T if hasattr(node, 'y_pre')], 1)
        self.g = T.stack([node.g for node in self.T if hasattr(node, 'g')], 1)

        return result
