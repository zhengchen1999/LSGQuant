import torch

from qdiff.base.base_quantizer import DynamicQuantizer


class DRAQQuantizer(DynamicQuantizer):
    """Dynamic-Range Adaptive activation Quantizer (DRAQ)."""

    def __init__(self, quant_config):
        super().__init__(quant_config)
        # per-channel dynamic range of the last call, kept for inspection
        self.channel_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize ``x`` of shape ``[N_group, C]`` (tokens x channels)."""
        assert len(x.shape) == 2, "DRAQ expects activations reshaped to [N_group, C]"

        # channel-wise dynamic range over all tokens: s_i = max_N |X_{N,i}|
        channel_scale = x.abs().amax(dim=0).clamp(1e-9).unsqueeze(0)  # [1, C]
        self.channel_scale = channel_scale

        # normalise to approximately [-1, 1]: X_tilde = X s^-1
        x_scaled = x / channel_scale
        assert not torch.isnan(x_scaled).any().item(), "nan exists in x"

        # symmetric per-token quantization, d_i = max_C |X_tilde_{i,C}|
        x_dequant = super().forward(x_scaled)

        # rescale to the original range
        return x_dequant * channel_scale
