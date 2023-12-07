import math
import numpy as np
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from functools import partial

from einops import rearrange

class BlockdiagMultiply(torch.autograd.Function):

    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        nblocks, q, p = weight.shape
        assert nblocks * p == n
        x_reshaped = x.reshape(batch_dim, nblocks, p).transpose(0, 1)
        out = torch.empty(batch_dim, nblocks, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out = torch.bmm(x_reshaped, weight.transpose(-1, -2), out=out).transpose(0, 1)
        return out.reshape(*batch_shape, nblocks * q)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        nblocks, q, p = weight.shape
        assert nblocks * p == n
        dx, dweight = None, None
        dout_reshaped = dout.reshape(batch_dim, nblocks, q).transpose(0, 1)
        if ctx.needs_input_grad[0]:
            dx = torch.empty(batch_dim, nblocks, p, device=x.device, dtype=x.dtype)
            dx = torch.bmm(dout_reshaped, weight.conj(),
                           out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
        if ctx.needs_input_grad[1]:
            x_reshaped = x.reshape(batch_dim, nblocks, p).transpose(0, 1)
            dweight = torch.bmm(dout_reshaped.transpose(-1, -2), x_reshaped.conj())
        return dx, dweight
blockdiag_multiply = BlockdiagMultiply.apply

class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """Subclasses should call reset_parameters
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Subclasses may override {in,out}_features_extended
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        self.set_weights_from_dense_init(dense_init_fn_=partial(init.kaiming_uniform_, a=math.sqrt(5)))
        self.reset_parameters_bias()

    def set_weights_from_dense_init(self, dense_init_fn_):
        raise NotImplementedError

    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in = self.bias.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def saving(self):
        raise NotImplementedError

    def convert_to_dense_weight(self):
        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}
        dense_weight = self.forward_matmul(torch.eye(self.in_features, **factory_kwargs)).T
        return dense_weight

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - in_features))
        return x

    def postprocess(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        # Convert bias to output.dtype in case of AMP, otherwise bias and activation will be in FP32
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output

class BlockdiagLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, shuffle=False, **kwargs):
        """shuffle: apply channel_shuffle operation before the matmul as in ShuffleNet
        """
        super().__init__(*args, **kwargs)
        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        self.shuffle = shuffle
        self.weight = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
        self.reset_parameters()

    def set_weights_from_dense_init(self, dense_init_fn_):
        dense_weight = torch.empty(self.out_features_extended, self.in_features_extended,
                                   device=self.weight.device, dtype=self.weight.dtype)
        dense_init_fn_(dense_weight)
        # Scale by sqrt because the weight is sparse
        scaling = math.sqrt(dense_weight.numel() / self.weight.numel())
        dense_weight *= scaling
        with torch.no_grad():
            nblocks = self.weight.shape[0]
            self.weight.copy_(rearrange(dense_weight, '(b o) (b1 i) -> b b1 o i',
                                        b=nblocks, b1=nblocks)[0])

    @property
    def saving(self):
        return self.weight.numel() / (self.in_features * self.out_features)

    def forward_matmul(self, x):
        x = self.preprocess(x)
        if self.shuffle:
            x = rearrange(x, '... (group c_per_group) -> ... (c_per_group group)',
                          group=self.weight.shape[0])  # group=nblocks
        output = blockdiag_multiply(x, self.weight)
        return self.postprocess(output)
