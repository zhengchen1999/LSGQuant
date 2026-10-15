import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.quant_layer import QuantizedLinear

class SQQuantizedLinear(QuantizedLinear):
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
        super().__init__(in_features, out_features, bias, device, quant_config, fp_module)

        self.alpha = quant_config.smooth_quant.alpha
        self.register_buffer('channel_mask', None)  # assigned outside, during PTQ 

    def get_channel_mask(self, act_mask):  # feed in the act channel-wise max mask
        # generate the weight mask
        weight_mask = self.fp_module.weight.abs().max(dim=0)[0] # [C_in]
        channel_mask = (weight_mask.abs()**self.alpha) / (act_mask.abs()**(1-self.alpha)) # negative value with **alpha will raise nan
        #print(f'weight={weight_mask.abs()[721]},act={act_mask.abs()[721]}')
        #print(f'channel_mask_shape={channel_mask.shape},nan={torch.isnan(channel_mask).any().item()}')
        self.channel_mask = channel_mask
        assert not torch.isinf(self.channel_mask).any().item(), "inf exists in channel_mask"

    def update_quantized_weight_scaled(self):
        assert self.channel_mask is not None
        C_out, C_in = self.fp_module.weight.shape
        #print(f"init={self.weight.data[0]}")
        self.w_quantizer.init_done = False
        self.weight.data = self.w_quantizer(self.fp_module.weight / self.channel_mask.reshape([1, C_in]))
        #print(f"updata={self.weight.data[0]}")
        assert not torch.isnan(self.weight.data).any().item(), "nan exists in weight"
        self.w_quantizer.init_done = True

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        input shape: [B,N_token,C]
        """
        if not self.quant_mode:  # use the FP
            return self.fp_module(x, *args, **kwargs)
        else:
            # reshape X into [G, -1] 
            B, N_token, C = x.shape
            x = x*self.channel_mask.reshape([1,1,C])

            x = x.reshape([B*N_token,-1])
            # isnan_mask = torch.isnan(x)
            # print(torch.nonzero(isnan_mask, as_tuple=False))
            assert not torch.isnan(x).any().item(), "nan exists in x"
            # quantize activation
            x = self.a_quantizer(x)
            x = x.reshape([B, N_token, C])

            # forward with dequantized weight and activation
            y = F.linear(x, self.weight, self.bias, *args, **kwargs)
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
        },
        'sq': {
            'alpha': 0.1,
        }
    }
    dummy_linear = nn.Linear(8,32, device='cuda')
    dummy_q_linear = SQQuantizedLinear(
        in_features = dummy_linear.in_features,
        out_features = dummy_linear.out_features,
        bias = dummy_linear.bias is not None,
        device = dummy_linear.weight.device,
        quant_config = dummy_q_config,
        fp_module = dummy_linear,
    )
    dummy_channel_mask = torch.rand([8], device='cuda')
    dummy_q_linear.channel_mask = dummy_channel_mask

    dummy_input = torch.rand([4,2048,8], device='cuda')
    output = dummy_q_linear(dummy_input)
