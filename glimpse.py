
import torch as T
import torch.nn.functional as F
import torch.nn as NN
from util import *

def gaussian_masks(c, d, s, len_, glim_len):
    '''
    c, d, s: 2D Tensor (batch_size, n_glims)
    len_, glim_len: int
    returns: 4D Tensor (batch_size, n_glims, glim_len, len_)
        each row is a 1D Gaussian
    '''
    batch_size, n_glims = c.size()

    # The original HART code did not shift the coordinates by
    # glim_len / 2.  The generated Gaussian attention does not
    # correspond to the actual crop of the bbox.
    # Possibly a bug?
    R = tovar(T.arange(0, glim_len).view(1, 1, 1, -1) - glim_len / 2)
    C = T.arange(0, len_).view(1, 1, -1, 1)
    C = C.expand(batch_size, n_glims, len_, 1)
    C = tovar(C)
    c = c[:, :, None, None]
    d = d[:, :, None, None]
    s = s[:, :, None, None]

    cr = c + R * d
    #sr = tovar(T.ones(cr.size())) * s
    sr = s

    mask = C - cr
    mask = (-0.5 * (mask / sr) ** 2).exp()

    mask = mask / (mask.sum(2, keepdim=True) + 1e-8)
    return mask


def extract_gaussian_glims(x, a, glim_size):
    '''
    x: 4D Tensor (batch_size, nchannels, nrows, ncols)
    a: 3D Tensor (batch_size, n_glims, att_params)
        att_params: (cx, cy, dx, dy, sx, sy)
    returns:
        5D Tensor (batch_size, n_glims, nchannels, n_glim_rows, n_glim_cols)
    '''
    batch_size, n_glims, _ = a.size()
    cx, cy, dx, dy, sx, sy = T.unbind(a, -1)
    _, nchannels, nrows, ncols = x.size()
    n_glim_rows, n_glim_cols = glim_size

    # (batch_size, n_glims, nrows, n_glim_rows)
    Fy = gaussian_masks(cy, dy, sy, nrows, n_glim_rows)
    # (batch_size, n_glims, ncols, n_glim_cols)
    Fx = gaussian_masks(cx, dx, sx, ncols, n_glim_cols)

    # (batch_size, n_glims, 1, nrows, n_glim_rows)
    Fy = Fy.unsqueeze(2)
    # (batch_size, n_glims, 1, ncols, n_glim_cols)
    Fx = Fx.unsqueeze(2)

    # (batch_size, 1, nchannels, nrows, ncols)
    x = x.unsqueeze(1)
    # (batch_size, n_glims, nchannels, n_glim_rows, n_glim_cols)
    g = Fy.transpose(-1, -2) @ x @ Fx

    return g

softplus_zero = F.softplus(tovar([0]))

class GaussianGlimpse(NN.Module):
    att_params = 6

    def __init__(self, glim_size):
        NN.Module.__init__(self)
        self.glim_size = glim_size

    @classmethod
    def full(cls):
        return tovar([0.5, 0.5, 1, 1, 0.5, 0.5])

    @classmethod
    def rescale(cls, x):
        y = [
                F.sigmoid(x[..., 0]),    # cx
                F.sigmoid(x[..., 1]),    # cy
                #F.softplus(x[..., 2]) / softplus_zero,   #dx
                #F.softplus(x[..., 3]) / softplus_zero,   #dy
                #F.softplus(x[..., 4]) / softplus_zero * 0.5, # sx
                #F.softplus(x[..., 5]) / softplus_zero * 0.5, # sy
                F.sigmoid(x[..., 2]) * 2,
                F.sigmoid(x[..., 3]) * 2,
                F.sigmoid(x[..., 4]),
                F.sigmoid(x[..., 5]),
                #x[..., 2].exp(),
                #x[..., 3].exp(),
                #x[..., 4].exp() * 0.5,
                #x[..., 5].exp() * 0.5,
                ]
        return T.stack(y, -1)

    @classmethod
    def absolute_to_relative(cls, att, relative):
        C_x, C_y, D_x, D_y, S_x, S_y = T.unbind(relative, -1)
        c_x, c_y, d_x, d_y, s_x, s_y = T.unbind(att, -1)
        return T.stack([
            (c_x - C_x) / D_x + 0.5,
            (c_y - C_y) / D_y + 0.5,
            d_x / D_x,
            d_y / D_y,
            s_x / D_x,
            s_y / D_y,
            ], -1)

    @classmethod
    def relative_to_absolute(cls, att, relative):
        C_x, C_y, D_x, D_y, S_x, S_y = T.unbind(relative, -1)
        c_x, c_y, d_x, d_y, s_x, s_y = T.unbind(att, -1)
        return T.stack([
            (c_x - 0.5) * D_x + C_x,
            (c_y - 0.5) * D_y + C_y,
            d_x * D_x,
            d_y * D_y,
            s_x * D_x,
            s_y * D_y
            ], -1)

    def forward(self, x, spatial_att):
        '''
        x: 4D Tensor (batch_size, nchannels, n_image_rows, n_image_cols)
        spatial_att: 3D Tensor (batch_size, n_glims, att_params) relative scales
        '''
        # (batch_size, n_glims, att_params)
        absolute_att = self._to_absolute_attention(spatial_att, x.size()[-2:])
        glims = extract_gaussian_glims(x, absolute_att, self.glim_size)

        return glims

    def att_to_bbox(self, spatial_att, x_size):
        '''
        spatial_att: (..., 6) [cx, cy, dx, dy, sx, sy] relative scales ]0, 1[
        return: (..., 4) [cx, cy, w, h] absolute scales
        '''
        cx = spatial_att[..., 0] * x_size[1]
        cy = spatial_att[..., 1] * x_size[0]
        w = T.abs(spatial_att[..., 2]) * (x_size[1] - 1)
        h = T.abs(spatial_att[..., 3]) * (x_size[0] - 1)
        bbox = T.stack([cx, cy, w, h], -1)
        return bbox

    def bbox_to_att(self, bbox, x_size):
        '''
        bbox: (..., 4) [cx, cy, w, h] absolute scales
        return: (..., 6) [cx, cy, dx, dy, sx, sy] relative scales ]0, 1[
        '''
        cx = bbox[..., 0] / x_size[1]
        cy = bbox[..., 1] / x_size[0]
        dx = bbox[..., 2] / (x_size[1] - 1)
        dy = bbox[..., 3] / (x_size[0] - 1)
        sx = bbox[..., 2] * 0.5 / x_size[1]
        sy = bbox[..., 3] * 0.5 / x_size[0]
        spatial_att = T.stack([cx, cy, dx, dy, sx, sy], -1)

        return spatial_att

    def _to_axis_attention(self, image_len, glim_len, c, d, s):
        c = c * image_len
        d = d * (image_len - 1) / (glim_len - 1)
        s = (s + 1e-5) * image_len / glim_len
        return c, d, s

    def _to_absolute_attention(self, params, x_size):
        '''
        params: 3D Tensor (batch_size, n_glims, att_params)
        '''
        n_image_rows, n_image_cols = x_size
        n_glim_rows, n_glim_cols = self.glim_size
        cx, dx, sx = T.unbind(params[..., ::2], -1)
        cy, dy, sy = T.unbind(params[..., 1::2], -1)
        cx, dx, sx = self._to_axis_attention(
                n_image_cols, n_glim_cols, cx, dx, sx)
        cy, dy, sy = self._to_axis_attention(
                n_image_rows, n_glim_rows, cy, dy, sy)

        # ap is now the absolute coordinate/scale on image
        # (batch_size, n_glims, att_params)
        ap = T.stack([cx, cy, dx, dy, sx, sy], -1)
        return ap


glimpse_table = {
        'gaussian': GaussianGlimpse,
        }
def create_glimpse(name, size):
    return glimpse_table[name](size)
