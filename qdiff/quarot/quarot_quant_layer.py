import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.quarot.quarot_utils import hadamard_from_sign, random_hadamard_sign

class QuarotQuantizedLinear(QuantizedLinear):
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

        self.rotation_matrix = None
        self.had_sign = None   # compact (+/-1) description of the Hadamard rotation
        self.device = device

    def get_rotation_matrix(self, device="cuda"):
        self.had_sign = random_hadamard_sign(self.in_features)
        self.rotation_matrix = hadamard_from_sign(self.had_sign).to(device)
        print(self.rotation_matrix.shape, self.rotation_matrix.dtype)

    def update_quantized_weight_rotated(self):
        self.w_quantizer.init_done = False   # unset the init done to overwrite quant_params
        self.weight.data = self.w_quantizer(torch.matmul(self.fp_module.weight.data, self.rotation_matrix.to(self.device)).float())
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
            x = torch.matmul(x, self.rotation_matrix.to(self.device)).to(dtype=x.dtype)
            x = x.reshape([B*N_token,-1])

            # quantize activation
            x = self.a_quantizer(x)
            x = x.reshape([B, N_token, C])

            y = F.linear(x, self.weight.to(x.dtype), self.bias, *args, **kwargs)

            return y
