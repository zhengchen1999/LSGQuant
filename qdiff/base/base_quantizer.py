import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import time
import math
from omegaconf import ListConfig

logger = logging.getLogger(__name__)

class BaseQuantizer(nn.Module):

    def __init__(self, quant_config):
        super(BaseQuantizer, self).__init__()

        # unpack the quant configurations
        self.n_bits = quant_config['n_bits']
        # self.group = quant_config['group']
        self.sym = quant_config.get('sym', False)

        if isinstance(self.n_bits, list):
            raise AssertionError("when multiple n_bits are adopted, use the MixedPrecisionBaseQuantizer")
        # assert self.group in ['token','tensor','channel']

        self.register_buffer('delta', None)
        self.register_buffer('zero_point', None)

        # INFO: for mixed_precision, the n_bits could be a ListConfig, and need to be initialized in subclass init
        if not isinstance(self.n_bits, ListConfig):
            self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1

        self.init_done = False

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("should be implemented in subclass.")

    def init_quant_params(self, x):
        raise NotImplementedError("should be implemented in subclass.")

    def extra_repr(self) -> str:
        base_str = super().extra_repr()
        return base_str + f"n_bits={self.n_bits}, symmetric={self.sym}"

class StaticQuantizer(BaseQuantizer):
    """
    the input shape should be [Group,-1]
    store the quant params (delta, zp) offline with init_quant_params
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)

        if self.sym:
            self.x_absmax = None
        else:
            self.x_max = None
            self.x_min = None

    def forward(self, x: torch.Tensor):
        x_quant = self.quantize(x)
        x_dequant = (x_quant + self.zero_point) * self.delta
        return x_dequant

    def quantize(self, x: torch.Tensor):

        if self.init_done is not True:  # set as True in ptq.py
            self.init_quant_params(x)
        x_int = torch.round(x / self.delta) - self.zero_point
        x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        return x_quant

    def init_quant_params(self, x):

        assert len(x.shape) == 2  # [N_group, -1]
        if self.sym:
            x_absmax = x.abs().max(dim=1)[0]
            self.x_absmax = (torch.max(self.x_absmax, x_absmax) if self.x_absmax is not None else x_absmax).to("cuda")  # update
            delta = x_absmax / self.n_levels
            zero_point = torch.zeros_like(delta, device=delta.device)
        else:
            x_max = x.max(dim=1)[0]
            x_max[x_max<0] = 0. 
            # sometimes the weight are init on CPU, but new data on GPU needed for update quant_params (quarot)
            self.x_max = torch.max(self.x_max.to(x_max.device), x_max) if self.x_max is not None else x_max

            x_min = x.min(dim=1)[0]
            x_min[x_min>0] = 0.
            self.x_min = torch.min(self.x_min.to(x_min.device), x_min) if self.x_min is not None else x_min

            delta = (x_max - x_min)/(self.n_levels-1)
            zero_point = torch.round(x_min/delta) + (self.n_levels/2)

        eps = 1.e-8
        try:
            assert torch.all(delta > eps), "unexpected small delta exists"
        except:
            # import ipdb; ipdb.set_trace()
            delta[delta < eps] = eps
            logger.info("unexpected small delta: {:.3e} exists in {}, set as eps".format(delta.abs().min(), ""))
            print("unexpected small delta: {:.3e} exists in {}, set as eps".format(delta.abs().min(), delta.abs().max()))

        self.delta = delta.unsqueeze(-1)  # [G] -> [G,1]
        self.zero_point = zero_point.unsqueeze(-1)

class DynamicQuantizer(BaseQuantizer):
    """
    the input shape should be [Group,-1]
    store the quant params (delta, zp) offline with init_quant_params
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)

    def quantize(self, x:torch.Tensor):
         # get the quant_params online
        assert len(x.shape) == 2  # [N_group, -1]
        assert torch.isnan(x).sum() == 0  # no nan exists

        if self.sym:
            x_absmax = x.abs().max(dim=1)[0]
            self.x_absmax = x_absmax

            delta = x_absmax / self.n_levels
            zero_point = torch.zeros_like(delta, device=delta.device)

            eps = 1.e-6
            try:
                assert torch.all(delta.abs() > eps)
            except:
                # import ipdb; ipdb.set_trace()
                delta[delta < eps] = eps
                logger.info("unexpected small delta: {:.3e} exists in {}, set as eps".format(delta.abs().min(), self.module_name))

        else:
            x_max = x.max(dim=1)[0]
            x_max[x_max<0] = 0. 
            self.x_max = x_max

            x_min = x.min(dim=1)[0]
            x_min[x_min>0] = 0.
            self.x_min = x_min

            delta = (x_max - x_min)/(self.n_levels-1)
            # INFO: check small values for delta, close to zero delta, would cause nan in zero_point
            eps = 1.e-8
            try:
                assert torch.all(delta.abs() > eps)
            except:
                import ipdb; ipdb.set_trace()

                delta[delta < eps] = eps
                logger.info("unexpected small delta: {:.3e} exists in {}, set as eps".format(delta.abs().min(), self.module_name))
            zero_point = torch.round(x_min/delta) + (self.n_levels/2)

        self.delta = delta.unsqueeze(-1)  # [G] -> [G,1]
        self.zero_point = zero_point.unsqueeze(-1)

        # quantize model with quant params
        x_int = torch.round(x / self.delta) - self.zero_point
        x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        return x_quant

    def forward(self, x: torch.Tensor):
        x_quant = self.quantize(x)
        x_dequant = (x_quant + self.zero_point) * self.delta
        return x_dequant

    def forward_with_quant_params(self, x, delta, mixed_precision=None):
        # INFO: used for attn block-wise quant, with precomputed delta
        # take in the x and delta with the same shape
        assert self.sym

        # INFO: meant to check attn_map only, but we use this for qk quant pre_softmax also 
        # try:
            # assert x.min()>=0 and x.max()<=1   # attn_map: the input is within [0,1] attn_map«
        # except:
            # import ipdb; ipdb.set_trace()

        if mixed_precision is not None:
            n_levels = torch.pow(2,mixed_precision) -  1 # 8bit: -> 255
            # aditional handling of 0-bit, since divide by 0 cause na
            zero_bit_mask = (n_levels != 0).int()
            n_levels[n_levels == 0] = 255  # temporarily set as 8-bit, masked anyway

        # INFO: check abnormally small delta_
        eps = 1.e-6
        try:
            assert torch.all(delta.abs() > eps)
        except:
            # import ipdb; ipdb.set_trace()  
            # safe to set it is eps.
            delta[delta < eps] = eps
            # logger.info("unexpected small delta: {:.3f} exists in attn_map, set as eps".format(delta.abs().min()))

        if mixed_precision is not None:
            delta = delta / n_levels
            x_int = torch.round(x / delta)
            # INFO: the torch.clamp takes single max value, but we want the same shape as x
            x_quant = torch.where(x_int>n_levels, n_levels, x_int)
        else:
            delta = delta/ (self.n_levels*2+1)
            x_int = torch.round(x / delta)
            x_quant = torch.clamp(x_int, 0, self.n_levels*2+1)

        x_dequant = (x_quant) * delta

        if mixed_precision is not None:  # apply the mask of elements of 0-bit
            x_dequant = x_dequant*zero_bit_mask

        return x_dequant
