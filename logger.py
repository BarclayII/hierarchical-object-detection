import logging
import torch as T
import torch.nn as NN
from functools import partial

debug = logging.Logger('debug')
debug.setLevel('DEBUG')

debug_handler = logging.FileHandler('debug.log', mode='w')
debug_fmt = logging.Formatter()
debug_handler.setFormatter(debug_fmt)
debug.addHandler(debug_handler)

def grad_summary(gs):
    return ' '.join(
            '(%.4f %.4f %.4f)' % (g.data.min(), g.data.max(), g.data.norm())
            for g in gs if g is not None
            )

def log_grad(module, grad_input, grad_output, name=None):
    grad_input_summary = grad_summary(grad_input)
    grad_output_summary = grad_summary(grad_output)
    #grad_params_summary = grad_summary(module.parameters())
    grad_params_summary = None
    debug.debug(
            '%s Grad Input: %s Grad Output: %s Grad Params: %s' % (
                name,
                grad_input_summary,
                grad_output_summary,
                grad_params_summary,
                )
            )

def register_backward_hooks(module):
    for name, obj in module.named_children():
        print(name)
        if isinstance(obj, NN.Module):
            obj.register_backward_hook(
                    partial(
                        log_grad,
                        name='%s.%s' % (type(module), name)
                        )
                    )
