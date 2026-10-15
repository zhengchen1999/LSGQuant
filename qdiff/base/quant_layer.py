import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer
from qdiff.base.mixed_precision_quantizer import MixedPrecisionStaticQuantizer, MixedPrecisionDynamicQuantizer
from omegaconf import ListConfig

class QuantizedLinear(torch.nn.Linear):
    """
    the base quantized linear layer,
    adpot the static weight quantization,
    and the dynamic activation quantization.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device: None,
        quant_config: dict,
        fp_module: torch.nn.Linear,
    ) -> None:
        super().__init__(in_features, out_features, bias, device)

        self.fp_module = fp_module
        self.q_cfg = quant_config

        # set default as None, to skip quant some part
        self.w_quantizer = None
        self.a_quantizer = None

        if quant_config.get('weight', None) is not None:
            w_quantizer_class = StaticQuantizer

            if isinstance(quant_config.weight.n_bits, ListConfig):  # mixed precision
                w_quantizer_class = MixedPrecisionStaticQuantizer

            self.w_quantizer = w_quantizer_class(quant_config['weight'])
            # INFO: the weights are initialized.
            # quantize the weight from FP module, bias remain as FP16
            self.weight.data = self.w_quantizer(fp_module.weight)  # [C_out, C_in]
            self.w_quantizer.init_done = True
        else:
            self.weight.data = fp_module.weight

        self.fp_weight = self.fp_module.weight
        self.bias = fp_module.bias

        if quant_config.get('act', None) is not None:
            if isinstance(quant_config.act.n_bits, ListConfig):
                self.a_quantizer = MixedPrecisionDynamicQuantizer(quant_config['act'])
            else:
                self.a_quantizer = DynamicQuantizer(quant_config['act'])

        self.use_kernel = False  # whether use the cuda kernel for actual saving
        self.quant_mode = True   # when set as False, use the original model forward

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        input shape: [B,N_token,C]
        """
        # print(f"weight shape: {self.weight.shape}")
        # print(f"bias shape: {self.bias.shape}")
        # print(f"x.shape: {x.shape}")
        if not self.quant_mode:  # use the FP
            return self.fp_module(x, *args, **kwargs)
        else:
            # reshape X into [G, -1]
            B, N_token, C = x.shape
            x = x.reshape([B*N_token,-1])

            # quantize activation
            if self.a_quantizer is not None:
                x = self.a_quantizer(x)
            x = x.reshape([B, N_token, C])

            # forward with dequantized weight and activation
            y = F.linear(x, self.weight, self.bias, *args, **kwargs)            
            y = y.reshape(B, N_token, -1)

            return y

if __name__ == '__main__':
    dummy_q_config = {
        'weight': {
            'n_bits': 8,
            'group': 'tensor',
            'sym': False
        },
        'act': {
            'n_bits': 8,
            'group': 'tensor',
            'sym': False
        }
    }
    dummy_linear = nn.Linear(8,32, device='cuda')
    dummy_q_linear = QuantizedLinear(
        in_features = dummy_linear.in_features,
        out_features = dummy_linear.out_features,
        bias = dummy_linear.bias is not None,
        device = dummy_linear.weight.device,
        quant_config = dummy_q_config,
        fp_module = dummy_linear,
    )
    dummy_input = torch.rand([4,2048,8], device='cuda')
    output = dummy_q_linear(dummy_input)
