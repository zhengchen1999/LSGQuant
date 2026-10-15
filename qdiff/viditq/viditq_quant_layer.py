import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.quarot.quarot_utils import hadamard_from_sign, random_hadamard_sign


class ViDiTQuantizedLinear(QuantizedLinear):
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

        self.alpha = quant_config.viditq.alpha
        self.register_buffer('channel_mask', None)  # assigned outside, during PTQ 
        self.rotation_matrix = None   # init so could be load in quant_params
        self.had_sign = None          # compact (+/-1) description of the Hadamard rotation
        self.device = device

    def get_channel_mask(self, act_mask):  # feed in the act channel-wise max mask
        # generate the weight mask
        weight_mask = self.fp_module.weight.abs().max(dim=0)[0] # [C_in]
        channel_mask = (weight_mask.abs()**self.alpha) / (act_mask.abs()**(1-self.alpha)) # negative value with **alpha will raise nan
        self.channel_mask = channel_mask
        assert not torch.isnan(channel_mask.any()), "nan exists in channel mask"

    def get_rotation_matrix(self, device):
        self.had_sign = random_hadamard_sign(self.in_features)
        self.rotation_matrix = hadamard_from_sign(self.had_sign).to(device)
        print(self.rotation_matrix.shape, self.rotation_matrix.dtype)

    def get_fake_rotation_matrix(self, device):
        self.rotation_matrix = torch.eye(n = self.in_features).to(dtype = torch.bfloat16, device = device)

    def get_fake_channel_mask(self, act_mask):
        _, C_in = self.fp_module.weight.shape
        self.channel_mask = torch.ones(C_in).to(dtype = torch.bfloat16, device = "cuda")

    def update_quantized_weight_rotated_and_scaled(self):

        # INFO: apply the scaling first, the apply rotation
        assert self.channel_mask is not None
        C_out, C_in = self.fp_module.weight.shape
        self.w_quantizer.init_done = False   # unset the init done to overwrite quant_params

        self.weight.data = self.w_quantizer(self.fp_module.weight / self.channel_mask.reshape([1, C_in]))
        self.weight.data = self.w_quantizer(torch.matmul(self.weight.data, self.rotation_matrix.to(self.device)).float())

        self.w_quantizer.init_done = True

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        input shape: [B,N_token,C]
        """
        # print(f"weight shape: {self.weight.shape}")
        # print(f"bias shape: {self.bias.shape}")
        # print(f"x.shape: {x.shape}")
        # print(f"channel_mask shape: {self.channel_mask.shape}")
        # print(f"rotation_matrix shape: {self.rotation_matrix.shape}")
        if not self.quant_mode:  # use the FP
            return self.fp_module(x, *args, **kwargs)
        else:
            # reshape X into [G, -1] 
            dtype_ = x.dtype
            B, N_token, C = x.shape
            x = x*self.channel_mask.reshape([1,1,C])  # first process through scale
            x = torch.matmul(x, self.rotation_matrix.to(self.device))  # then rotate
            x = x.reshape([B*N_token,-1])

            # quantize activationq
            x = self.a_quantizer(x)
            x = x.reshape([B, N_token, C])

            # forward with dequantized weight and activation
            y = F.linear(x, self.weight.to(dtype=dtype_), self.bias, *args, **kwargs)

            return y
